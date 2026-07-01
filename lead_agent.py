"""
Autonomous Lead Generation Agent (LangGraph + Groq + DuckDuckGo)
==================================================================

Give it a plain-English request like:

    "find all scientists working on agentic AI these days"

...and it will:
  1. Search the web for relevant pages (labs, faculty pages, papers, teams).
  2. Visit the most promising pages and read them.
  3. Pull out real people's names, organizations, and emails.
  4. Keep looping (search -> scrape -> search again) until it has a good list.
  5. Save everything to leads.csv (Name, Email, Organization, Source URL).

This is a genuine ReAct-style LangGraph agent: the LLM itself decides which
tool to call and when to stop, using LangGraph's prebuilt `tools_condition`
routing (not a hardcoded search -> extract -> save pipeline).

Author: generated for Qamar
"""

import os
import re
import csv
import sys
import time
from typing import List

from dotenv import load_dotenv

load_dotenv()  # reads GROQ_API_KEY etc. from a local .env file if present

import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition


# ---------------------------------------------------------------------------
# 0. Config
# ---------------------------------------------------------------------------

# openai/gpt-oss-120b has noticeably more reliable native tool-calling on
# Groq than llama-3.3-70b-versatile, which occasionally emits malformed
# "<function=...>" text instead of a proper tool call. You can still
# override via the GROQ_MODEL env var if you want to experiment.
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
OUTPUT_CSV = os.environ.get("LEADS_OUTPUT_CSV", "leads.csv")
MAX_AGENT_STEPS = int(os.environ.get("MAX_AGENT_STEPS", "40"))

EMAIL_REGEX = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")


# ---------------------------------------------------------------------------
# 1. Tools
#    (Tool descriptions ARE the prompt the LLM sees when deciding what to
#    call, so each one is written to be specific and unambiguous.)
# ---------------------------------------------------------------------------

@tool
def web_search(query: str, max_results: int = 8) -> str:
    """Search the public web for pages related to a topic, field, or role.

    Use this FIRST to discover candidate pages that likely list real
    people's names and contact info: university faculty/team pages,
    research lab "People" pages, personal academic websites, conference
    speaker lists, company "About us" pages, recent papers with author
    affiliations, etc.

    Args:
        query: A focused search query. Prefer specific phrasing such as
            "<topic> research lab team" or "<topic> faculty directory"
            or "<topic> researchers 2026" rather than a raw copy of the
            user's full sentence.
        max_results: How many results to fetch (default 8, max 15).

    Returns:
        A numbered list of results, each with a title, URL, and short
        snippet. Use the URLs with the scrape_website tool.
    """
    max_results = max(1, min(max_results, 15))
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        return f"SEARCH_ERROR: {e}"

    if not results:
        return "NO_RESULTS: try a different / broader query."

    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "").strip()
        href = r.get("href", "").strip()
        body = r.get("body", "").strip()
        lines.append(f"{i}. {title}\n   URL: {href}\n   Snippet: {body}")
    return "\n\n".join(lines)


@tool
def scrape_website(url: str) -> str:
    """Fetch a specific web page and read its visible text content.

    Use this on URLs returned by web_search (or on other URLs you find
    while reading pages, e.g. a "Contact" or "Team" link) to look for a
    person's full name, title/organization, and email address.

    Args:
        url: The exact page URL to fetch. Must start with http:// or https://

    Returns:
        The page's visible text (truncated to a reasonable length) plus a
        separate "EMAILS_FOUND_ON_PAGE" line listing any email addresses
        detected on the page via pattern matching. If no emails are
        detected, that line will say "none".
    """
    if not url.startswith(("http://", "https://")):
        return "SCRAPE_ERROR: url must start with http:// or https://"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        )
    }
    try:
        resp = requests.get(url, headers=headers, timeout=12)
        resp.raise_for_status()
    except Exception as e:
        return f"SCRAPE_ERROR: could not fetch {url} ({e})"

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = re.sub(r"\n{2,}", "\n", soup.get_text("\n")).strip()
    emails = sorted(set(EMAIL_REGEX.findall(resp.text)))

    text_truncated = text[:4000]
    emails_line = ", ".join(emails) if emails else "none"

    return (
        f"PAGE_TEXT (truncated):\n{text_truncated}\n\n"
        f"EMAILS_FOUND_ON_PAGE: {emails_line}"
    )


class SaveLeadsInput(BaseModel):
    # IMPORTANT: kept intentionally FLAT (single string field) rather than
    # List[SomeNestedModel]. Groq's constrained tool-calling grammar for
    # Llama models can break on nested list-of-object schemas and emit a
    # malformed "<function=...>" string instead of a valid tool call,
    # which raises BadRequestError: tool_use_failed. A flat JSON-string
    # argument avoids that failure mode entirely.
    leads_json: str = Field(
        description=(
            "A JSON array (as a STRING) of lead objects gathered so far. "
            "Each object must have exactly these keys: 'name', 'email', "
            "'organization', 'source_url'. Example: "
            '[{"name": "Jane Doe", "email": "jane@example.com", '
            '"organization": "MIT CSAIL", "source_url": "https://..."}]. '
            "If no email was found for someone, use \"not_found\". "
            "If organization is unknown, use \"unknown\"."
        )
    )


@tool(args_schema=SaveLeadsInput)
def save_leads(leads_json: str) -> str:
    """FINAL STEP ONLY. Save the complete list of collected leads to a CSV file.

    Call this exactly ONCE, only after you are done searching and scraping,
    passing a JSON-array STRING of every real lead you found across all
    pages (deduplicated by name+email). Do not call this early with a
    partial list, and do not call it more than once.

    Only include entries with a real, specific person's full name (not a
    lab name, not a generic department). If you found a name but no email,
    still include the row with email set to "not_found".
    """
    import json as _json

    try:
        rows = _json.loads(leads_json)
    except Exception as e:
        return (
            f"SAVE_ERROR: leads_json was not valid JSON ({e}). "
            "Retry by passing a valid JSON array string, e.g. "
            '[{"name": "...", "email": "...", "organization": "...", "source_url": "..."}]'
        )

    if not isinstance(rows, list):
        return "SAVE_ERROR: leads_json must be a JSON array of objects."

    required_keys = {"name", "email", "organization", "source_url"}
    seen = set()
    unique_rows = []
    for r in rows:
        if not isinstance(r, dict) or not required_keys.issubset(r.keys()):
            continue
        key = (str(r["name"]).strip().lower(), str(r["email"]).strip().lower())
        if key in seen:
            continue
        seen.add(key)
        unique_rows.append(
            {
                "name": r["name"],
                "email": r["email"],
                "organization": r["organization"],
                "source_url": r["source_url"],
            }
        )

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["name", "email", "organization", "source_url"]
        )
        writer.writeheader()
        writer.writerows(unique_rows)

    return f"Saved {len(unique_rows)} unique leads to {OUTPUT_CSV}."


TOOLS = [web_search, scrape_website, save_leads]


# ---------------------------------------------------------------------------
# 2. LLM + system prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an autonomous Lead Generation Research Agent.

The user will describe a topic, field, or type of role (for example:
"scientists working on agentic AI" or "startups doing solar panel recycling").
Your job is to find REAL people currently active in that space and collect
their Name, Email, Organization, and the Source URL where you found them.

You have three tools: web_search, scrape_website, and save_leads.

Follow this loop:
1. Call web_search with a focused query to find pages likely to list real
   people (lab "team" pages, university faculty directories, recent papers,
   conference speaker pages, company "about us" pages).
2. For each promising URL in the results, call scrape_website to read it and
   look for full names, organizations, and email addresses.
3. If a page names people but has no email, try another web_search such as
   "<person name> <organization> email" or "<person name> contact page",
   then scrape_website that result too.
4. Keep searching and scraping with varied queries until you have gathered
   as many distinct, real people as you reasonably can (aim for at least
   8-15 if the topic has enough coverage; fewer is fine if the topic is
   niche - never invent people to hit a number).
5. When you are done gathering, call save_leads exactly once with the full,
   deduplicated list.

Hard rules:
- NEVER invent, guess, or hallucinate a name or email. Only use what tools
  actually returned. If you cannot find an email for someone, set it to
  "not_found" but still include their name and organization.
- Do not call save_leads more than once, and do not call it until you have
  finished researching.
- Prefer primary sources (university/lab/company pages) over aggregator or
  social-media snippets.
- Keep working autonomously - do not ask the user clarifying questions;
  make reasonable interpretations of the request yourself.
"""


def build_llm():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Get a free key at https://console.groq.com/keys "
            "and set it as an environment variable (or in a .env file)."
        )
    return ChatGroq(model=GROQ_MODEL, temperature=0.1, api_key=api_key)


# ---------------------------------------------------------------------------
# 3. Graph (real conditional routing, not a fixed 3-step pipeline)
# ---------------------------------------------------------------------------

def build_graph():
    llm = build_llm()
    llm_with_tools = llm.bind_tools(TOOLS)

    def agent_node(state: MessagesState) -> dict:
        # Groq's tool-calling grammar occasionally slips and emits a
        # malformed "<function=...>" string instead of a real tool call,
        # which raises BadRequestError (code: tool_use_failed). Rather than
        # letting one bad generation kill the whole run, retry a few times
        # with a short nudge appended, then fall back to a plain (no-tool)
        # call so the agent can at least tell the user what went wrong.
        from groq import BadRequestError

        messages = state["messages"]
        last_error = None
        for attempt in range(3):
            try:
                response = llm_with_tools.invoke(messages)
                return {"messages": [response]}
            except BadRequestError as e:
                last_error = e
                nudge = HumanMessage(
                    content=(
                        "Your last response used an invalid tool-call format. "
                        "Call a tool using the proper structured tool-calling "
                        "interface only - do not write '<function=...>' as text."
                    )
                )
                messages = messages + [nudge]
                time.sleep(1)

        # All retries failed - degrade gracefully instead of crashing.
        from langchain_core.messages import AIMessage

        return {
            "messages": [
                AIMessage(
                    content=(
                        "I hit a repeated tool-calling error from the model "
                        f"provider and had to stop early ({last_error}). "
                        "Try re-running the cell, or switch GROQ_MODEL to "
                        "a different tool-calling model."
                    )
                )
            ]
        }

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))

    graph.add_edge(START, "agent")
    # tools_condition inspects the last AI message: if it contains tool
    # calls -> route to "tools", otherwise -> route to END.
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile()


# ---------------------------------------------------------------------------
# 4. Runner
# ---------------------------------------------------------------------------

def run(request: str, verbose: bool = True) -> str:
    """Run the agent on a natural-language request and return the CSV path."""
    app = build_graph()

    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=request)]

    final_state = None
    for step in app.stream(
        {"messages": messages},
        {"recursion_limit": MAX_AGENT_STEPS},
        stream_mode="values",
    ):
        final_state = step
        if verbose:
            last = step["messages"][-1]
            role = last.__class__.__name__
            tool_calls = getattr(last, "tool_calls", None)
            if tool_calls:
                for tc in tool_calls:
                    print(f"[{role}] -> calling tool: {tc['name']}({tc['args']})")
            else:
                content = str(getattr(last, "content", ""))[:300]
                if content:
                    print(f"[{role}] {content}")

    if os.path.exists(OUTPUT_CSV):
        return OUTPUT_CSV
    return "(agent finished but did not call save_leads - see log above)"


if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_request = " ".join(sys.argv[1:])
    else:
        user_request = input("What kind of leads do you want to find? ")

    print(f"\nRunning lead generation agent for: {user_request!r}\n")
    start = time.time()
    result = run(user_request)
    print(f"\nDone in {time.time() - start:.1f}s -> {result}")
