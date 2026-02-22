import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { SSEClientTransport } from "@modelcontextprotocol/sdk/client/sse.js";
// qmd-mcp 伺服器的 HTTP 端點
const transport = new SSEClientTransport(new URL("http://127.0.0.1:8181/mcp"));
const client = new Client({ name: "test-client", version: "1.0.0" }, { capabilities: {} });

async function main() {
  await client.connect(transport);
  const result = await client.callTool({
    name: "search",
    arguments: {
      query: "如何安裝 Member GRPC SDK?"
    }
  });
  console.log(JSON.stringify(result, null, 2));
  process.exit(0);
}

main().catch(err => {
  console.error("Error:", err);
  process.exit(1);
});
