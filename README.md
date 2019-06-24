
# Fleet
**Fleet** is High-Level API for distributed training in PaddlePaddle. The name of **Fleet** means that a large crowd of ships working together to finish a large scale job. The design of **Fleet** makes a trade-off between easy-to-use and algorithmic extensibility. First, a user can shift from single machine paddle fluid code to distributed code within ten lines of code. Second, different algorithms can be easily defined through distributed strategy through **Fleet** API.

# Design of Fleet

## Role Maker
A **Role Maker** specifies distributed node role in a distributed training job. For example, in parameter server training scenario, a **Role Maker** appoints current node as a worker or a server, and total node number of current distributed job will be available in **Role Maker**. Currently supported **Role Makers** are as follows:
- MPISymetricRoleMaker
- UserDefinedRoleMaker
- PaddleCloudRoleMaker

## Fleet Mode
A **Fleet** API is available in https://github.com/PaddlePaddle/Paddle, a user can easily import different modes of Fleet APIk. Current available **Fleet Mode** are as follows:
- PSLib Mode
- Distribute Transpiler Mode
- Collective Mode

## Quick Start

### PSLib Mode
```
import incubate fleet
```

### Distribute Transpiler Mode
```
import distribute_transpiler as fleet
```

### Collective Mode
```
import collective as fleet
```

# Performance Evaluation


