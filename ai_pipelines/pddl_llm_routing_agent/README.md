This PDDL domain, problem, and plan setup is intended to be used with a ReAct-style agent for intelligent routing of requests across multiple LLM providers. The intended approach would leverage the following components:

- **MCP-based Tool Calling**: The agent uses Model-Call Planning (MCP) to select and invoke LLMs as tools, based on the requirements of each incoming request.
- **Unified Planning with PDDL**: The agent formulates the routing decision as a planning problem, encoding the available LLMs, their capabilities, provider constraints, and request requirements in PDDL. The Unified Planning framework, together with a numeric-fluent planner (e.g., lpg-td), is used to generate optimal or feasible plans for request routing.
- **ReAct-style Reasoning**: The agent interleaves retrieval of relevant OpenRouter documentation (covering model capabilities, pricing, context limits, etc.) with planning and action execution. Retrieved docs are used to ground the agent's decisions, ensuring that routing choices are explainable and context-aware.
- **Dynamic Decision-Making**: For each request, the agent retrieves the relevant documentation sections, encodes them as PDDL facts (e.g., which LLM covers which section, provider costs, context window limits), and updates the problem file. The planner then computes a plan that selects the most suitable LLM provider, possibly considering cost, capability, and other constraints.
- **Plan Execution**: The resulting plan specifies which LLM/provider to use for the request, and the agent executes the plan by invoking the selected LLM via tool-calling APIs.

This setup enables flexible, data-driven, and explainable routing of requests in multi-provider LLM environments, with decisions grounded in up-to-date documentation and formalized via automated planning.



In order to run domain.pddl and problem.pddl with a planner you will need to run this command:

    planutils run lpg-td -- \
    pddl_llm_routin_agent/domain.pddl \
    pddl_llm_routin_agent/problem.pddl \
    -n 1 \
    > pddl_llm_routin_agent/plan.out




REQUIREMENTS (for running on WSL):
________________________________________

| **planutils** | Install with 'rye add planutiles'
| **Singularity / Apptainer** | planutils runs planners inside .sif images. Use this command to install on WSL `sudo apt update && sudo apt install -y singularity-container` |

# numeric-fluent planner
planutils install lpg-td  