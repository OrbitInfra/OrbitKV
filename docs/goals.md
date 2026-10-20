# Overview

OrbitKV is an external KV cache for vLLM and SGLang. It retains reusable prefixes
in DRAM and optional SSD so requests can recover state after GPU eviction or an
engine restart. Run an independent Cache Manager per inference host and connect
the official engine adapter. Begin with the [quickstart](single-node.md).

## What you can use today

The starting profile is dense TP=1/PP=1 inference on official vLLM 0.31.0 or
SGLang 0.5.22. Engines own HBM and scheduling; OrbitKV owns external replicas,
query budgets and transfer completion. Raw DRAM Restore executes in the engine;
Publish and SSD/codec work execute in the Manager.

Optional peer caching uses Manager inventory streams for local discovery and
Mooncake TENT for data transfer. Native P/D uses the engines' own transports with
independent cache adapters. Check [compatibility and support](compatibility.md)
before selecting a model, topology or feature.

## Ownership and compatibility

Cache identity binds model artifacts, computation and byte layout. Compiled
recovery validates adapter-declared page/window/checkpoint requirements; it does
not analyze arbitrary model graphs or predict future tokens. A shared API does
not make different engines' KV bytes interchangeable. Source and destination
holds remain owned through transfer completion.

## Experimental and planned work

Shared-cache HA, complete hybrid external recovery, public SGLang lifecycle
integration and broader parallelism retain separate gates. Native GDS and
physical transfer lifetime remain unqualified. General joint retention/placement
planning is future work; limited observations and guarded choices already exist.

The [architecture](architecture.md) explains contracts, [performance](performance.md)
separates measurements from support claims, and the [completion plan](completion-plan.md)
defines the remaining execution sequence. A workload with little reuse may pay
extra copy/storage cost; size and measure the cache against your own traffic.
