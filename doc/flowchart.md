# JUST Exposure Time Calculator (ETC) Workflow

This compact flowchart illustrates the core physical modeling process within the JUST ETC engine, tracing from inputs to the final Signal-to-Noise Ratio calculation.

```mermaid
graph LR
    %% Define styles
    classDef input fill:#f9f2f4,stroke:#d0a0b0,stroke-width:1.5px,color:#333;
    classDef calc fill:#e8f4f8,stroke:#8fbcd4,stroke-width:1.5px,color:#333;
    classDef result fill:#e6f9ec,stroke:#8cc69f,stroke-width:1.5px,color:#333;

    %% External Inputs
    Target["Target Parameters<br/>(SED, z, Mag, Reff)"]:::input
    Obs["Obs. Conditions<br/>(Seeing, Airmass, Moon)"]:::input
    Inst["Instrument Config<br/>(Aperture, THRPUT, CCD)"]:::input

    %% Processing Pipeline
    Target --> Norm["Normalize Flux<br/>(F_&nu;)"]:::calc
    
    subgraph Engine ["ETC Core Physical Pipeline (Per Wavelength)"]
        direction LR
        Norm --> Atten["Galactic & Atmospheric<br/>Extinction"]:::calc
        Atten --> Optics["Telescope Collection<br/>& Vignetting"]:::calc
        Optics --> Geo["Fiber Geometric<br/>Throughput"]:::calc
        Geo --> Thr["Instrument Throughput<br/>& Trace Extraction"]:::calc
        Thr --> Sig["Source Signal<br/>(e-)"]:::result
        
        Obs -.-> Bkg["Sky Background<br/>(Continuum + Lines)"]:::calc
        Bkg --> Noise["Total Noise Variance<br/>(Sky+Dark+Read+Sys)"]:::result
        Inst -.-> Noise
        Sig -. Poisson .-> Noise
    end

    %% Routing inputs to relevant processes
    Obs -.-> Atten
    Obs -.-> Geo
    Inst -.-> Optics
    Inst -.-> Thr

    %% Final Outputs
    Sig --> SNR["Compute SNR<br/>(compute_snr)"]:::result
    Noise --> SNR
    SNR --> Solver["Solve Exposure Time<br/>(solve_exposure_time)"]:::result
```
