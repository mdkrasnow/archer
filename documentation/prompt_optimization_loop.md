```mermaid
flowchart TB
    %% Main States
    Start([Start])
    InitSystem[Initialize Archer System]
    InitPrompts[Set Initial Prompts]
    ForwardPass[Run Forward Pass]
    HumanValidation[Human Validation]
    BackwardPass[Run Backward Pass]
    GenerateVariants[Generate Prompt Variants]
    EvaluateCandidates[Evaluate Prompt Candidates]
    SelectBest[Select Top Prompts]
    UpdateActive[Update Active Prompts]
    CheckConvergence{Converged?}
    End([End])
    
    %% Flow
    Start --> InitSystem
    InitSystem --> InitPrompts
    InitPrompts --> ForwardPass
    
    %% Forward Pass Flow
    ForwardPass --> |Generated Content and AI Evaluations| HumanValidation
    HumanValidation --> |Validated Evaluations| BackwardPass
    
    %% Backward Pass Flow
    BackwardPass --> GenerateVariants
    GenerateVariants --> |Prompt Candidates| EvaluateCandidates
    EvaluateCandidates --> |Evaluated Candidates| SelectBest
    SelectBest --> UpdateActive
    UpdateActive --> CheckConvergence
    
    %% Convergence Decision
    CheckConvergence --> |Yes| End
    CheckConvergence --> |No| ForwardPass
    
    %% Detailed Forward Pass
    subgraph ForwardPassDetail[Forward Pass Details]
        direction TB
        InputProcess[Process Input Data]
        GenerateContent[Generate Content with Prompts]
        EvaluateContent[AI Evaluates Generated Content]
        StoreEvaluations[Store Evaluations]
        
        InputProcess --> GenerateContent
        GenerateContent --> EvaluateContent
        EvaluateContent --> StoreEvaluations
    end
    
    %% Detailed Backward Pass
    subgraph BackwardPassDetail[Backward Pass Details]
        direction TB
        AnalyzeEval[Analyze Evaluations]
        IdentifyIssues[Identify Issues in Prompts]
        OptimizePrompt[Optimize Prompts]
        GenerateVariations[Generate Variations]
        
        AnalyzeEval --> IdentifyIssues
        IdentifyIssues --> OptimizePrompt
        OptimizePrompt --> GenerateVariations
    end
    
    %% Detailed Evaluation
    subgraph EvaluationDetail[Prompt Evaluation Details]
        direction TB
        SimulateInputs[Simulate Inputs]
        TestPromptPerformance[Test Prompt Performance]
        ScorePrompts[Score Prompts]
        RankPrompts[Rank Prompts]
        
        SimulateInputs --> TestPromptPerformance
        TestPromptPerformance --> ScorePrompts
        ScorePrompts --> RankPrompts
    end
    
    %% Connect Subgraphs
    ForwardPass -.-> ForwardPassDetail
    BackwardPass -.-> BackwardPassDetail
    EvaluateCandidates -.-> EvaluationDetail
    
    %% Styling
    classDef process fill:#bfb,stroke:#333,stroke-width:2px
    classDef decision fill:#fbf,stroke:#333,stroke-width:2px
    classDef subgraph fill:#bbf,stroke:#333,stroke-width:2px
    classDef endpoint fill:#fbb,stroke:#333,stroke-width:2px
    
    class InitSystem,InitPrompts,ForwardPass,HumanValidation,BackwardPass,GenerateVariants,EvaluateCandidates,SelectBest,UpdateActive process
    class CheckConvergence decision
    class ForwardPassDetail,BackwardPassDetail,EvaluationDetail subgraph
    class Start,End endpoint
``` 