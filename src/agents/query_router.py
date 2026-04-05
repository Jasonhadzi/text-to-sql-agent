"""Query Router Agent — dynamically routes to correct tables/data source.

Given a clear question, reads the schema config and decides which tables
are relevant. Returns a list of relevant tables to the NLQ Agent.

# TODO: Rachel — implement agent, must read from config/datasource_config.json dynamically
"""

query_router_agent = None
