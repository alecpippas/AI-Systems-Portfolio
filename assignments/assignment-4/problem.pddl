(define (problem openrouter-requests) (:domain openrouter)
    (:objects
        ; Providers
        openai anthropic google        - provider

        ; LLMs
        gpt4o gpt3dot5 claude2 gemini  - llm

        ; Billing accounts
        acct1 acct2                   - account

        ; Incoming requests
        req1 req2                     - request
    )


    (:init
        ;todo: put the initial state's facts and numeric values here
        ; which provider offers which model
        (provided-by gpt4o    openai)
        (provided-by gpt3dot5 openai)
        (provided-by claude2  anthropic)
        (provided-by gemini   google)

        ; which models cover which document sections
        (covers gpt4o    quickstart)
        (covers gpt4o    simple_agentic_loop)
        (covers gpt3dot5 quickstart)
        (covers claude2  simple_agentic_loop)

        ; each request’s topic requirement
        (requires req1 quickstart)
        (requires req2 simple_agentic_loop)

        ; billing-account <--> request
        (account-of req1 acct1)
        (account-of req2 acct2)

        ; initialize numeric fluents: budgets, context‐limits, costs, token‐needs
        (= (remaining-budget acct1)    10)
        (= (remaining-budget acct2)    10)

        (= (context-limit    gpt4o)    2000)
        (= (context-limit    gpt3dot5) 1000)
        (= (context-limit    claude2)  1500)
        (= (context-limit    gemini)   800)

        (= (required-context req1)     500)
        (= (required-context req2)     1200)

        (= (cost-per-token   gpt4o)    0.005)
        (= (cost-per-token   gpt3dot5) 0.01)
        (= (cost-per-token   claude2)  0.02)
        (= (cost-per-token   gemini)   0.015)
    )

    (:goal 
        (and ;todo: put the goal condition here
            (routed req1 gpt3dot5)  ; cheapest model covering quickstart under acct1’s budget
            (routed req2 gpt4o)     ; only gpt4o covers the agentic-loop topic under acct2’s limits
        )
    )


    (:metric minimize 0)           ; trivial metric – nothing to optimise - makes compatible with metric-ff planner's cost-minimizing search algo
)
