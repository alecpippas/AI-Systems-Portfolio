;Header and description

(define (domain openrouter)

    ;remove requirements that are not needed
    (:requirements :typing :adl :negative-preconditions :fluents :action-costs)

    (:types ;todo: enumerate types and their hierarchy here, e.g. car truck bus - vehicle
    llm 
    provider 
    request 
    account 
    section
    )

    ; un-comment following line if constants are needed
    (:constants
        quickstart
        faq_getting_started
        faq_support
        faq_billing_usage
        models_and_providers
        api_technical_specifications
        privacy_data_logging
        credit_billing_systems
        account_management
        principles
        model_browser
        model_variants
        model_routing
        provider_routing
        prompt_caching
        structured_outputs
        tool_function_calling
        simple_agentic_loop
        - section
    )  ; all of these are of type section
    ; constants resemble the top 15-level topics extracted from OpenRouter docs

    (:predicates
        ; which provider supplies which model
        (provided-by      ?l - llm    ?p - provider)

        ; which model “covers” ( is expert in) each doc‐section
        (covers           ?l - llm    ?s - section)

        ; which incoming request requires which section
        (requires         ?r - request  ?s - section)

        ; billing account tied to each request
        (account-of       ?r - request  ?a - account)
        
        ; whether we’ve routed request ?r to model ?l
        (routed           ?r - request  ?l - llm)
    )


    (:functions ; define numeric fluents (functions) here, technically its a state variable that holds a numeric value that can change from step to step
        (remaining-budget ?a - account)    ; how much $ left
        (context-limit    ?l - llm    )    ; max tokens LLM can handle
        (required-context ?r - request )   ; tokens needed by request
        (cost-per-token   ?l - llm    )    ; $ per token for this LLM
    )
    (:action route-request
        :parameters (?r - request ?l - llm ?a - account ?s - section)
        :precondition (and
            (requires         ?r ?s)
            (covers           ?l ?s)
            (account-of       ?r ?a)
            (not (routed      ?r ?l))
            (>= (remaining-budget ?a)
                (* (cost-per-token ?l)
                    (required-context ?r)))
            (>= (context-limit ?l)
                (required-context ?r))
        )
        :effect (and
            (routed ?r ?l)
            (decrease (remaining-budget ?a)
                    (* (cost-per-token ?l)
                        (required-context ?r)))
        )
    )
)