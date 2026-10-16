SGM runtime [ms] over 20 instances using DPLG-N. Bold: fastest variant. Speedup: B&B over AF+AT. Cert.: AF+AT nodes as a percentage of AF nodes.

| Problem | Rule | B&B | AT | AF | AF+AT | Gurobi | Speedup | Cert. [%] |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Diagnostic | MF | 1163.1 | 233.6 | 1191.3 | **13.5** | 486.7 | 86.2× | 0.9 |
|  | PC | 1237.2 | 17.7 | 1249.7 | **15.2** | 486.7 | 81.7× | 0.9 |
| Integrator | MF | 154.4 | 158.9 | 155.6 | **2.5** | 28.0 | 60.6× | 1.8 |
|  | PC | 44.5 | 46.5 | 43.6 | **3.0** | 28.0 | 14.8× | 6.9 |
| Buck | MF | 527.7 | 534.4 | 537.2 | **7.7** | 191.3 | 68.9× | 1.1 |
|  | PC | 323.1 | 328.1 | 328.7 | **7.9** | 191.3 | 40.6× | 1.7 |
| Refrigeration | MF | 107.3 | 110.1 | 111.5 | **8.5** | 168.1 | 12.6× | 5.5 |
|  | PC | 83.4 | 83.3 | 84.4 | **8.6** | 168.1 | 9.7× | 8.2 |
