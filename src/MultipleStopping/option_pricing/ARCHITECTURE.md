# Evaluation Framework Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        OPTION PRICING FRAMEWORK                      │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  1. PATH GENERATION (Extensible)                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│    PathGenerator (ABC)                                               │
│         ├── GBMPathGenerator                                         │
│         ├── JumpDiffusionPathGenerator                               │
│         └── HistoricalDataPathGenerator                              │
│                                                                       │
│    All environments and algorithms can use any generator             │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  2. ENVIRONMENT                                                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│    BermudanPutEnv(params, generator, seed)                           │
│         - Accepts any PathGenerator                                  │
│         - Pre-generates full path at reset()                         │
│         - Gym-style interface for RL training                        │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  3. PRICING METHODS (Trained independently)                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐│
│    │   LSM Algorithm │    │  Q-Learning     │    │      DQN        ││
│    │                 │    │                 │    │                 ││
│    │  - Regression   │    │  - Q-table      │    │  - Neural net   ││
│    │  - Backward     │    │  - Epsilon-     │    │  - Experience   ││
│    │    induction    │    │    greedy       │    │    replay       ││
│    │                 │    │  - Discretized  │    │  - Continuous   ││
│    └────────┬────────┘    └────────┬────────┘    └────────┬────────┘│
│             │                      │                       │         │
│             └──────────────────────┴───────────────────────┘         │
│                                    │                                 │
│                         Generate policy_fn(s, t)                     │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘

                                    │
                                    ▼

┌─────────────────────────────────────────────────────────────────────┐
│  4. EVALUATION FRAMEWORK (Fair Comparison)                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│    TestSet(params, n_paths=5000, generator, seed=999)               │
│         │                                                             │
│         ├── Generate FIXED test paths (once)                         │
│         │   - Same paths for ALL methods                             │
│         │   - Eliminates simulation noise                            │
│         │                                                             │
│         ├── evaluate_policy(policy_fn, name, time)                  │
│         │   └──> EvaluationResult                                    │
│         │        ├── method_name                                     │
│         │        ├── mean_value                                      │
│         │        ├── std_error                                       │
│         │        ├── values                                          │
│         │        ├── exercise_times                                  │
│         │        └── training_time                                   │
│         │                                                             │
│         └── evaluate_longstaff_schwartz()                           │
│             └──> EvaluationResult                                    │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘

                                    │
                                    ▼

┌─────────────────────────────────────────────────────────────────────┐
│  5. COMPARISON & VISUALIZATION                                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│    ComparisonVisualizer(results: List[EvaluationResult], params)    │
│         │                                                             │
│         ├── print_summary_table()                                   │
│         │   - Formatted comparison table                             │
│         │   - Gaps from benchmark                                    │
│         │   - All key metrics                                        │
│         │                                                             │
│         └── plot_summary()                                          │
│             ├── Price comparison (bar chart)                         │
│             ├── Exercise time distributions                          │
│             ├── Value distributions                                  │
│             ├── Training time comparison                             │
│             ├── Exercise statistics (box plots)                      │
│             └── Summary table                                        │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘


WORKFLOW EXAMPLE:
═════════════════

    1. Create test set (ONCE)
       ┌────────────────────────────────────┐
       │ test_set = TestSet(params,         │
       │   n_paths=5000, seed=999)          │
       └────────────────────────────────────┘
                      │
                      ▼
    2. Evaluate each method
       ┌────────────────────────────────────┐
       │ lsm_result = test_set              │
       │   .evaluate_longstaff_schwartz()   │
       ├────────────────────────────────────┤
       │ ql_result = test_set               │
       │   .evaluate_policy(ql_policy,      │
       │     "Q-Learning", time)            │
       ├────────────────────────────────────┤
       │ dqn_result = test_set              │
       │   .evaluate_policy(dqn_policy,     │
       │     "DQN", time)                   │
       └────────────────────────────────────┘
                      │
                      ▼
    3. Compare all methods
       ┌────────────────────────────────────┐
       │ results = [lsm_result,             │
       │            ql_result,               │
       │            dqn_result]              │
       ├────────────────────────────────────┤
       │ viz = ComparisonVisualizer(        │
       │   results, params)                 │
       ├────────────────────────────────────┤
       │ viz.print_summary_table()          │
       │ viz.plot_summary()                 │
       └────────────────────────────────────┘


KEY BENEFITS:
═════════════

✓ Same paths for all methods    → Fair comparison
✓ Organized results             → Clean code
✓ Reusable components           → DRY principle
✓ Easy to extend                → Add methods in 3 lines
✓ Comprehensive metrics         → Full analysis
✓ Production-ready              → Type-safe, documented



EXTENSIBILITY:
══════════════

Adding a new method:

    # 1. Train your agent
    my_agent = MyAgent(params)
    start = time.time()
    my_agent.train(env, n_episodes=1000)
    training_time = time.time() - start
    
    # 2. Get policy function
    my_policy = my_agent.get_policy_function()
    
    # 3. Evaluate on test set (3 lines!)
    my_result = test_set.evaluate_policy(my_policy, "My Method", training_time)
    
    # 4. Add to comparison
    all_results.append(my_result)
    viz = ComparisonVisualizer(all_results, params)
    viz.plot_summary()
    
    # Done! Your method is now fully integrated!
```
