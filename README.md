# HSLU · AI Agents and Controlled Workflows: public course files

Public parts of Module 3 of the CAS LLMs and AI Agents at HSLU (Lucerne University of Applied Sciences and Arts).
The course notebooks install the helper package from here.

| Folder | What it is |
|---|---|
| `claims-lab/` | The `claims_lab` Python package: synthetic data for the fictional insurer Lakeside Insurance AG, tools, MCP servers, scoreboard, reference solutions and the replay recording for the demo cells |
| `claude-desktop/` | How to run the package's MCP servers in Claude Desktop on your computer (notebook 03, at home) |

All data is invented for teaching. Any resemblance to real persons or businesses is coincidental.

```python
!pip install -q "claims-lab @ git+https://github.com/alexander-stanenko/hslu-public.git#subdirectory=claims-lab"
import claims_lab as cl
cl.setup()
```

The package needs an API key for the course model (Azure OpenAI or Anthropic), read from Colab Secrets or the environment.
`CLAIMS_LAB_PROVIDER=offline` runs it with a scripted stand-in model, without a key and at no cost.
