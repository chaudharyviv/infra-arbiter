"""
Domain registry for the Enterprise Infrastructure Advisor.

Each domain defines:
- unit: the capacity unit the sidebar slider represents
- workloads: {name: {benchmark metrics, search keywords}}
- vendors: {name: {strengths, workloads, deployment, cost_profile, min_capacity, ...}}
- tco: deterministic 3-yr TCO parameters for TCOEngine
  (list_rates per unit by cost profile, cloud monthly tiers, facilities per unit)

The LangGraph workflow is domain-agnostic - adding a new domain (e.g. Network,
Backup Software) is purely a data change here.
"""

DOMAINS = {
    # ======================================================
    "Storage": {
        "unit": "TB",
        "capacity_slider": {"label": "Capacity (TB)", "min": 10, "max": 2000, "default": 500, "step": 50},
        "workloads": {
            "OLTP Database": {"metrics": {"IOPS": "50K-500K", "Latency": "<500µs"},
                              "keywords": "database storage low latency IOPS all-flash array"},
            "AI/ML Training": {"metrics": {"IOPS": "100K-1M", "Latency": "<200µs"},
                               "keywords": "AI ML GPU storage parallel filesystem Weka BeeGFS VAST high throughput"},
            "HPC": {"metrics": {"IOPS": "100K-2M", "Latency": "<100µs"},
                    "keywords": "HPC parallel filesystem Lustre GPFS BeeGFS DDN Exascaler WekaFS"},
            "VDI": {"metrics": {"IOPS": "30K-200K", "Latency": "<1ms"},
                    "keywords": "VDI virtual desktop storage IOPS boot storm"},
            "Virtualization": {"metrics": {"IOPS": "30K-100K", "Latency": "<1ms"},
                               "keywords": "virtualization VMware storage vSAN hyper-converged"},
            "File Services": {"metrics": {"IOPS": "10K-50K", "Latency": "<2ms"},
                              "keywords": "file storage NAS SMB NFS scale-out"},
            "Backup": {"metrics": {"IOPS": "5K-20K", "Latency": "<5ms"},
                       "keywords": "backup storage object storage S3 data protection deduplication"},
        },
        "vendors": {
            "Pure Storage": {"strengths": ["Performance", "Simplicity", "Data Reduction"],
                             "workloads": ["OLTP Database", "VDI", "AI/ML Training", "Virtualization"],
                             "deployment": ["on-prem", "hybrid"], "cost_profile": "premium",
                             "sweet_spot": "50TB-500TB", "min_capacity": 20},
            "NetApp": {"strengths": ["Multi-cloud", "Data Management", "Data Protection"],
                       "workloads": ["OLTP Database", "File Services", "Virtualization", "Backup"],
                       "deployment": ["on-prem", "hybrid", "cloud"], "cost_profile": "competitive",
                       "sweet_spot": "100TB-2PB", "min_capacity": 10},
            "Dell": {"strengths": ["Scale", "Integration", "Cost-Effective"],
                     "workloads": ["OLTP Database", "HPC", "Virtualization", "Backup"],
                     "deployment": ["on-prem", "hybrid"], "cost_profile": "competitive",
                     "sweet_spot": "100TB-2PB", "min_capacity": 10},
            "HPE": {"strengths": ["Composability", "HCI", "Predictive Analytics"],
                    "workloads": ["OLTP Database", "VDI", "Virtualization", "Backup"],
                    "deployment": ["on-prem", "hybrid"], "cost_profile": "competitive",
                    "sweet_spot": "50TB-1PB", "min_capacity": 10},
            "AWS": {"strengths": ["Scale", "Cloud-native Services", "Global Infrastructure"],
                    "workloads": ["OLTP Database", "AI/ML Training", "Backup", "File Services", "HPC"],
                    "deployment": ["cloud"], "cost_profile": "variable",
                    "services": {"block": "EBS, FSx for ONTAP", "object": "S3", "file": "EFS, FSx for Lustre"},
                    "min_capacity": 1},
            "Azure": {"strengths": ["Enterprise Integration", "AI Services", "Hybrid Cloud"],
                      "workloads": ["OLTP Database", "AI/ML Training", "Virtualization", "File Services", "Backup"],
                      "deployment": ["cloud"], "cost_profile": "variable",
                      "services": {"block": "Managed Disks, ANF", "object": "Blob", "file": "Azure Files"},
                      "min_capacity": 1},
            "GCP": {"strengths": ["AI/ML", "Analytics", "Global Network"],
                    "workloads": ["AI/ML Training", "HPC", "Backup"],
                    "deployment": ["cloud"], "cost_profile": "variable",
                    "services": {"block": "Persistent Disk, Hyperdisk", "object": "Cloud Storage", "file": "Filestore, Parallelstore"},
                    "min_capacity": 1},
            "VAST Data": {"strengths": ["AI/ML Performance", "Scale-out", "All-Flash"],
                          "workloads": ["AI/ML Training", "HPC"],
                          "deployment": ["on-prem", "hybrid"], "cost_profile": "premium",
                          "sweet_spot": "500TB-10PB", "min_capacity": 100},
            "Weka": {"strengths": ["Ultra Performance", "Cloud-native", "GPU-Optimized"],
                     "workloads": ["AI/ML Training", "HPC"],
                     "deployment": ["on-prem", "cloud", "hybrid"], "cost_profile": "premium",
                     "sweet_spot": "100TB-5PB", "min_capacity": 50},
            "IBM": {"strengths": ["Enterprise Grade", "Mainframe Integration", "Reliability"],
                    "workloads": ["Backup", "File Services", "Virtualization"],
                    "deployment": ["on-prem", "hybrid", "cloud"], "cost_profile": "competitive",
                    "sweet_spot": "100TB-5PB", "min_capacity": 10},
            "Cohesity": {"strengths": ["Data Protection", "Simplicity", "Ransomware Defense"],
                         "workloads": ["Backup", "File Services"],
                         "deployment": ["on-prem", "cloud", "hybrid"], "cost_profile": "competitive",
                         "sweet_spot": "50TB-2PB", "min_capacity": 20},
            "Huawei": {"strengths": ["Cost-Effective", "Scale", "Innovation"],
                       "workloads": ["OLTP Database", "HPC", "Virtualization", "File Services"],
                       "deployment": ["on-prem", "hybrid"], "cost_profile": "budget",
                       "sweet_spot": "100TB-5PB", "min_capacity": 10},
            "MinIO": {"strengths": ["S3-Compatible", "Open Source", "Multi-cloud"],
                      "workloads": ["Backup", "AI/ML Training", "File Services"],
                      "deployment": ["on-prem", "cloud", "hybrid"], "cost_profile": "budget",
                      "sweet_spot": "100TB-10PB", "min_capacity": 10},
            "DDN": {"strengths": ["Exascale Performance", "AI400X", "Parallel Filesystem"],
                    "workloads": ["HPC", "AI/ML Training"],
                    "deployment": ["on-prem", "hybrid"], "cost_profile": "premium",
                    "sweet_spot": "1PB-50PB", "min_capacity": 500},
        },
        "tco": {
            "list_rates": {"premium": 2600, "competitive": 1800, "budget": 1100},  # $/TB 3yr
            "cloud_monthly_tiers": [(100, 26.0), (500, 20.0), (float("inf"), 15.0)],  # $/TB-month
            "facilities_per_unit": 180,
            "migration_flat": 25000,
        },
    },

    # ======================================================
    "Server / Compute": {
        "unit": "servers",
        "capacity_slider": {"label": "Server Count", "min": 2, "max": 500, "default": 50, "step": 2},
        "workloads": {
            "Virtualization Hosts": {"metrics": {"vCPU density": "High", "Memory": "1-4TB/node"},
                                     "keywords": "virtualization hosts VMware ESXi Nutanix HCI servers"},
            "AI/GPU Compute": {"metrics": {"GPUs": "4-8/node", "Interconnect": "NVLink/IB"},
                               "keywords": "GPU servers AI training NVIDIA H100 HGX liquid cooling"},
            "HPC Compute": {"metrics": {"Cores": "96-192/node", "Interconnect": "InfiniBand"},
                            "keywords": "HPC compute nodes dense servers InfiniBand cluster"},
            "General Application": {"metrics": {"Cores": "32-64/node", "Memory": "256-512GB"},
                                    "keywords": "rack servers general purpose application hosting"},
            "Container Platform": {"metrics": {"Density": "High pod/node", "Memory": "512GB-1TB"},
                                   "keywords": "Kubernetes worker nodes container platform bare metal"},
            "Database Hosts": {"metrics": {"Memory": "1-6TB/node", "Storage": "NVMe local"},
                               "keywords": "database servers high memory NVMe mission critical"},
        },
        "vendors": {
            "Dell": {"strengths": ["PowerEdge Portfolio", "Supply Chain", "iDRAC Management"],
                     "workloads": ["Virtualization Hosts", "General Application", "Database Hosts", "Container Platform", "AI/GPU Compute"],
                     "deployment": ["on-prem", "hybrid"], "cost_profile": "competitive", "min_capacity": 2},
            "HPE": {"strengths": ["ProLiant/Synergy", "GreenLake Consumption", "iLO"],
                    "workloads": ["Virtualization Hosts", "General Application", "Database Hosts", "HPC Compute"],
                    "deployment": ["on-prem", "hybrid"], "cost_profile": "competitive", "min_capacity": 2},
            "Lenovo": {"strengths": ["ThinkSystem", "Price/Performance", "Reliability"],
                       "workloads": ["Virtualization Hosts", "General Application", "HPC Compute", "Container Platform"],
                       "deployment": ["on-prem", "hybrid"], "cost_profile": "budget", "min_capacity": 2},
            "Cisco UCS": {"strengths": ["Fabric Integration", "Intersight", "Blade Density"],
                          "workloads": ["Virtualization Hosts", "General Application", "Database Hosts"],
                          "deployment": ["on-prem", "hybrid"], "cost_profile": "premium", "min_capacity": 4},
            "Supermicro": {"strengths": ["GPU Density", "Time-to-Market", "Cost"],
                           "workloads": ["AI/GPU Compute", "HPC Compute", "Container Platform"],
                           "deployment": ["on-prem"], "cost_profile": "budget", "min_capacity": 2},
            "Nutanix (HCI)": {"strengths": ["HCI Simplicity", "1-Click Ops", "Hybrid Cloud"],
                              "workloads": ["Virtualization Hosts", "Container Platform", "General Application"],
                              "deployment": ["on-prem", "hybrid"], "cost_profile": "premium", "min_capacity": 3},
            "AWS EC2": {"strengths": ["Instance Breadth", "Autoscaling", "Spot Economics"],
                        "workloads": ["General Application", "Container Platform", "AI/GPU Compute", "Database Hosts", "HPC Compute"],
                        "deployment": ["cloud"], "cost_profile": "variable", "min_capacity": 1},
            "Azure VMs": {"strengths": ["Enterprise Integration", "Hybrid Benefit Licensing", "Confidential Compute"],
                          "workloads": ["General Application", "Virtualization Hosts", "Database Hosts", "Container Platform"],
                          "deployment": ["cloud"], "cost_profile": "variable", "min_capacity": 1},
            "GCP Compute Engine": {"strengths": ["Custom Machine Types", "TPU/GPU", "Live Migration"],
                                   "workloads": ["General Application", "AI/GPU Compute", "Container Platform", "HPC Compute"],
                                   "deployment": ["cloud"], "cost_profile": "variable", "min_capacity": 1},
        },
        "tco": {
            "list_rates": {"premium": 52000, "competitive": 34000, "budget": 22000},  # $/server 3yr incl support
            "cloud_monthly_tiers": [(20, 650.0), (100, 520.0), (float("inf"), 420.0)],  # $/server-equiv month
            "facilities_per_unit": 4500,   # power/cooling/rack per server 3yr
            "migration_flat": 40000,
        },
    },

    # ======================================================
    "Database": {
        "unit": "TB data",
        "capacity_slider": {"label": "Data Size (TB)", "min": 1, "max": 500, "default": 20, "step": 1},
        "workloads": {
            "OLTP / Core Banking": {"metrics": {"TPS": "10K-500K", "Latency": "<5ms"},
                                    "keywords": "OLTP database core banking ACID high availability"},
            "OLAP / Analytics": {"metrics": {"Scan": "TB/s class", "Concurrency": "High"},
                                 "keywords": "analytics data warehouse columnar OLAP MPP"},
            "In-Memory": {"metrics": {"Latency": "<1ms", "Memory": "TB-scale"},
                          "keywords": "in-memory database real-time SAP HANA Redis"},
            "NoSQL / Document": {"metrics": {"Scale": "Horizontal", "Schema": "Flexible"},
                                 "keywords": "NoSQL document database MongoDB scale-out"},
            "Time-Series / Telemetry": {"metrics": {"Ingest": "1M+ pts/s", "Retention": "Tiered"},
                                        "keywords": "time series database telemetry monitoring ingest"},
        },
        "vendors": {
            "Oracle": {"strengths": ["Exadata Performance", "RAC HA", "Enterprise Features"],
                       "workloads": ["OLTP / Core Banking", "OLAP / Analytics"],
                       "deployment": ["on-prem", "hybrid", "cloud"], "cost_profile": "premium", "min_capacity": 1},
            "Microsoft SQL Server": {"strengths": ["Ecosystem Integration", "Always On AG", "TCO vs Oracle"],
                                     "workloads": ["OLTP / Core Banking", "OLAP / Analytics"],
                                     "deployment": ["on-prem", "hybrid", "cloud"], "cost_profile": "competitive", "min_capacity": 1},
            "PostgreSQL (EDB)": {"strengths": ["Open Source", "Oracle Compatibility (EDB)", "No License Lock-in"],
                                 "workloads": ["OLTP / Core Banking", "OLAP / Analytics", "Time-Series / Telemetry"],
                                 "deployment": ["on-prem", "hybrid", "cloud"], "cost_profile": "budget", "min_capacity": 1},
            "SAP HANA": {"strengths": ["In-Memory Speed", "SAP Landscape Native", "Analytics"],
                         "workloads": ["In-Memory", "OLAP / Analytics"],
                         "deployment": ["on-prem", "hybrid", "cloud"], "cost_profile": "premium", "min_capacity": 1},
            "MongoDB": {"strengths": ["Developer Velocity", "Horizontal Scale", "Atlas Managed"],
                        "workloads": ["NoSQL / Document", "Time-Series / Telemetry"],
                        "deployment": ["on-prem", "cloud", "hybrid"], "cost_profile": "competitive", "min_capacity": 1},
            "Redis Enterprise": {"strengths": ["Sub-ms Latency", "Caching + Primary", "Active-Active Geo"],
                                 "workloads": ["In-Memory", "Time-Series / Telemetry"],
                                 "deployment": ["on-prem", "cloud", "hybrid"], "cost_profile": "competitive", "min_capacity": 1},
            "GCP Cloud SQL / AlloyDB": {"strengths": ["Managed PostgreSQL", "AlloyDB Columnar Engine", "IAM Integration"],
                                        "workloads": ["OLTP / Core Banking", "OLAP / Analytics"],
                                        "deployment": ["cloud"], "cost_profile": "variable", "min_capacity": 1},
            "AWS Aurora / RDS": {"strengths": ["Managed Scale", "Multi-AZ", "Serverless Option"],
                                 "workloads": ["OLTP / Core Banking", "NoSQL / Document"],
                                 "deployment": ["cloud"], "cost_profile": "variable", "min_capacity": 1},
            "Azure SQL / Cosmos DB": {"strengths": ["Hybrid Benefit", "Global Distribution (Cosmos)", "Managed Instance"],
                                      "workloads": ["OLTP / Core Banking", "NoSQL / Document"],
                                      "deployment": ["cloud"], "cost_profile": "variable", "min_capacity": 1},
        },
        "tco": {
            "list_rates": {"premium": 42000, "competitive": 18000, "budget": 7000},  # $/TB data 3yr (license+support+infra alloc)
            "cloud_monthly_tiers": [(10, 900.0), (50, 700.0), (float("inf"), 520.0)],  # $/TB-month managed
            "facilities_per_unit": 0,   # infra allocation already in rate
            "migration_flat": 60000,    # DB migrations are expensive
        },
    },

    # ======================================================
    "Middleware": {
        "unit": "instances",
        "capacity_slider": {"label": "Instance Count", "min": 2, "max": 200, "default": 20, "step": 2},
        "workloads": {
            "Messaging / Streaming": {"metrics": {"Throughput": "100K-10M msg/s", "Durability": "Replicated"},
                                      "keywords": "messaging streaming Kafka MQ event-driven banking"},
            "API Gateway": {"metrics": {"RPS": "10K-1M", "Latency": "<20ms added"},
                            "keywords": "API gateway management security rate limiting banking"},
            "Application Server": {"metrics": {"JVM/Runtime": "Managed", "HA": "Clustered"},
                                   "keywords": "application server Java EE JBoss WebSphere Tomcat"},
            "Integration / ESB": {"metrics": {"Connectors": "100+", "Patterns": "EIP"},
                                  "keywords": "integration ESB iPaaS MuleSoft TIBCO connectors"},
            "Caching Layer": {"metrics": {"Latency": "<1ms", "Hit Ratio": ">95%"},
                              "keywords": "distributed cache Redis Hazelcast in-memory grid"},
        },
        "vendors": {
            "IBM (MQ / WebSphere)": {"strengths": ["Banking Pedigree", "Transactional Guarantees", "z/OS Integration"],
                                     "workloads": ["Messaging / Streaming", "Application Server", "Integration / ESB"],
                                     "deployment": ["on-prem", "hybrid", "cloud"], "cost_profile": "premium", "min_capacity": 2},
            "Red Hat (JBoss / AMQ)": {"strengths": ["Open Source + Support", "OpenShift Native", "Cost vs Proprietary"],
                                      "workloads": ["Application Server", "Messaging / Streaming", "Integration / ESB", "Caching Layer"],
                                      "deployment": ["on-prem", "hybrid", "cloud"], "cost_profile": "competitive", "min_capacity": 2},
            "Confluent (Kafka)": {"strengths": ["Streaming Standard", "Connectors", "Exactly-Once"],
                                  "workloads": ["Messaging / Streaming"],
                                  "deployment": ["on-prem", "cloud", "hybrid"], "cost_profile": "premium", "min_capacity": 3},
            "Apache (Kafka/Tomcat/ActiveMQ)": {"strengths": ["Zero License Cost", "Community", "Ubiquity"],
                                               "workloads": ["Messaging / Streaming", "Application Server", "Caching Layer"],
                                               "deployment": ["on-prem", "cloud", "hybrid"], "cost_profile": "budget", "min_capacity": 1},
            "TIBCO": {"strengths": ["Enterprise Integration Depth", "BusinessWorks", "Low-Latency FTL"],
                      "workloads": ["Integration / ESB", "Messaging / Streaming"],
                      "deployment": ["on-prem", "hybrid"], "cost_profile": "premium", "min_capacity": 2},
            "MuleSoft": {"strengths": ["API-led Connectivity", "Anypoint Platform", "Salesforce Ecosystem"],
                         "workloads": ["Integration / ESB", "API Gateway"],
                         "deployment": ["cloud", "hybrid"], "cost_profile": "premium", "min_capacity": 2},
            "Google Apigee": {"strengths": ["API Analytics", "Monetization", "GCP Native"],
                              "workloads": ["API Gateway"],
                              "deployment": ["cloud", "hybrid"], "cost_profile": "premium", "min_capacity": 2},
            "Kong": {"strengths": ["Cloud-Native Gateway", "Performance", "Open Core"],
                     "workloads": ["API Gateway"],
                     "deployment": ["on-prem", "cloud", "hybrid"], "cost_profile": "competitive", "min_capacity": 2},
            "Hazelcast": {"strengths": ["In-Memory Grid", "Stream Processing", "Java Native"],
                          "workloads": ["Caching Layer", "Messaging / Streaming"],
                          "deployment": ["on-prem", "cloud", "hybrid"], "cost_profile": "competitive", "min_capacity": 2},
        },
        "tco": {
            "list_rates": {"premium": 58000, "competitive": 30000, "budget": 6000},  # $/instance 3yr (license+support)
            "cloud_monthly_tiers": [(10, 1400.0), (50, 1100.0), (float("inf"), 850.0)],  # $/instance-month managed
            "facilities_per_unit": 0,
            "migration_flat": 35000,
        },
    },
}

DEPLOYMENTS = ["On-Premises", "Cloud", "Hybrid"]


def get_matching_vendors(domain: str, workload: str, deployment: str, capacity: int = 0):
    """Domain-generic vendor matching (same semantics as v1 storage logic)."""
    cfg = DOMAINS[domain]
    deploy_key = deployment.lower()

    if "on-prem" in deploy_key:
        valid = {"on-prem", "hybrid"}
    elif "cloud" in deploy_key:
        valid = {"cloud"}
    elif "hybrid" in deploy_key:
        valid = {"on-prem", "cloud", "hybrid"}
    else:
        valid = {"on-prem"}

    matching = []
    for vendor, data in cfg["vendors"].items():
        if workload not in data.get("workloads", []):
            continue
        vendor_deployments = set(data.get("deployment", []))
        if deploy_key == "cloud" and "cloud" not in vendor_deployments:
            continue
        if not (vendor_deployments & valid):
            continue
        if capacity > 0 and capacity < data.get("min_capacity", 0):
            continue
        matching.append(vendor)
    return matching
