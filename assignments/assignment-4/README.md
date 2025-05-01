In order to run domain.pddl and problem.pddl with a planner you will need to run this command:

    planutils run lpg-td -- \
    assignment-4/domain.pddl \
    assignment-4/problem.pddl \
    -n 1 \
    > assignment-4/plan.out




REQUIREMENTS (for running on WSL):
________________________________________

| **planutils** | Install with 'rye add planutiles'
| **Singularity / Apptainer** | planutils runs planners inside .sif images. Use this command to install on WSL `sudo apt update && sudo apt install -y singularity-container` |

# numeric-fluent planner
planutils install lpg-td  