# The course's MCP servers in Claude Desktop

Notebook 03 connects an agent to two MCP servers from the claims-lab package. The same servers also run in Claude Desktop on your own computer:

| Server | Module in `claims_lab` | What it offers |
|---|---|---|
| `lakeside-claims-systems` | `mcp_servers.claims_systems` | five tools (`lookup_policy`, `find_policies`, `get_claim_history`, `search_claims_by_invoice`, `calc_payout`), the resource `policy://{policy_id}` and the prompt `review_claim` |
| `lakeside-policy-docs` | `mcp_servers.policy_docs` | the tool `search_terms` and the resource `terms://car` |

Claude Desktop starts both as local processes. They need no API key: the data is the fictional Lakeside database inside the package. Claude Desktop needs the internet for its model, and a Claude account.

Why not "Add custom connector" with a URL: Claude's custom connectors are reached from Anthropic's cloud, so the server would have to be reachable from the public internet. A server on your own computer goes into Claude Desktop's configuration file instead ([Anthropic help: local MCP servers](https://support.claude.com/en/articles/10949351-getting-started-with-local-mcp-servers-on-claude-desktop)).

Checked on 10 November 2026 with Claude Desktop 4.23118 on macOS.

## 1. A Python environment with the course package

Python 3.10 or newer. On macOS or Linux:
```bash
python3 -m venv ~/claims-lab-venv
```
```bash
~/claims-lab-venv/bin/pip install "claims-lab @ git+https://github.com/alexander-stanenko/hslu-public.git#subdirectory=claims-lab"
```
On Windows the interpreter is `%USERPROFILE%\claims-lab-venv\Scripts\python.exe`; use it wherever the steps below say `~/claims-lab-venv/bin/python`.

## 2. The two entries for Claude Desktop

The package prints them, with the paths of your environment:
```bash
~/claims-lab-venv/bin/python -c "from claims_lab import mcp_servers; mcp_servers.claude_desktop_config(['claims_systems', 'policy_docs'])"
```
The output starts with `"mcpServers"`. Each entry names a server and the command that starts it, plus a working folder (`~/claims_lab_work`) where the server keeps its copy of the database.

## 3. Add them to the configuration file

The file is `claude_desktop_config.json`:
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

Open it from Claude Desktop's settings or in a text editor. It already contains other keys, such as `"preferences"`. Add the printed `"mcpServers": { ... }` as one more top-level key, with a comma after its closing brace, and leave the rest as it is:
```json
{
  "mcpServers": {
    "lakeside-claims-systems": { "command": "...", "args": ["-m", "claims_lab.mcp_servers.claims_systems"], "env": { ... } },
    "lakeside-policy-docs": { "command": "...", "args": ["-m", "claims_lab.mcp_servers.policy_docs"], "env": { ... } }
  },
  "preferences": { ... }
}
```
If the file is empty or missing, it contains only the printed text inside one pair of braces.

Quit Claude Desktop completely (Cmd+Q on macOS; closing the window is not enough) and start it again.

## 4. Try it

In a new chat, click **+** below the message box, then **Connectors**: both Lakeside servers should be listed. Then ask: *Is windscreen damage covered for policy P-1001, and what is the deductible?*

Claude asks for permission before it runs `lookup_policy`, and perhaps `search_terms`. The answer should say that a broken windscreen counts as glass damage, which P-1001 covers, with a deductible of CHF 500. The wording varies from run to run.

Only add the two servers above. `claude_desktop_config()` without names also adds the course's `comms` server, whose tools send e-mail and record decisions.

## If it does not work

| Symptom | Likely cause | Fix |
|---|---|---|
| The servers are not listed under Connectors | Claude Desktop was not restarted, or the file is no longer valid JSON | Quit with Cmd+Q and restart; check the file with `python3 -m json.tool` followed by its path |
| A server is listed with an error | `command` does not point to the environment with claims-lab | Run `~/claims-lab-venv/bin/python -c "import claims_lab"`; on macOS, read the log in `~/Library/Logs/Claude/` (files starting with `mcp-server-lakeside-`) |
| Settings do not let you change the configuration | A managed (company) installation can switch off local MCP servers | Use a personal installation, or try the servers in the MCP Inspector (below) |

## Without Claude Desktop: the MCP Inspector

The Inspector is a developer tool that runs in the browser and needs Node.js:
```bash
cd ~ && npx @modelcontextprotocol/inspector ~/claims-lab-venv/bin/python -m claims_lab.mcp_servers.claims_systems
```
Click **Connect**, then **Tools**, **List Tools**. Choose `lookup_policy`, enter `P-1001` and click **Run Tool**.
