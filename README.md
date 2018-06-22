# model_task_allocation

This project includes the source code for simulations of the model of task allocation in social insects. 
There are two environmental types: static and dynamic, handled by separate files. 
In each file, there are two learning mechanisms involved: individual learning and social learning.

To run the simulation for static environments, the command "python simulation_static_environments.py JSON_FILE LEARNING_MODE" is needed. 
A sample json file with a set of parameter values is involved in this repository.

To run the simulation for dynamic environments, a direct call of the function specifying the target learning type is needed.
