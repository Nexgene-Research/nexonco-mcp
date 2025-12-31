# Smithery Deployment Guide

This guide explains how to deploy the Nexonco MCP server to [Smithery](https://smithery.ai), a platform for hosting MCP servers.

## Prerequisites

- [Smithery CLI](https://github.com/smithery-ai/cli) installed: `npm install -g @smithery/cli`
- Docker (for local testing)
- Smithery account

## Deployment Configuration

The repository is configured for Smithery deployment with the following files:

### smithery.yaml

The `smithery.yaml` file defines the deployment configuration:

```yaml
runtime: "container"
build:
  dockerfile: "Dockerfile"
  dockerBuildPath: "."
startCommand:
  type: "http"
  configSchema:
    type: "object"
    properties: {}
    additionalProperties: false
```

### Dockerfile

The Dockerfile has been configured to:
- Use the `PORT` environment variable (defaults to 8081 for Smithery)
- Expose the server on the configured port
- Run the MCP server with SSE (Server-Sent Events) transport

## Key Features for Smithery Compatibility

The server includes the following Smithery-compatible features:

1. **Configurable Port**: Listens on the `PORT` environment variable
2. **HTTP/SSE Transport**: Uses Server-Sent Events over HTTP
3. **CORS Support**: Enables cross-origin requests with proper headers
4. **Multiple Endpoints**:
   - `/mcp` - Smithery-compatible MCP endpoint
   - `/sse` - Original SSE endpoint
   - `/health` - Health check endpoint
   - `/version` - API version information
   - `/` - Homepage with documentation

## Deploying to Smithery

To deploy the server to Smithery:

```bash
# Navigate to the repository root
cd nexonco-mcp

# Deploy to Smithery
smithery deploy .
```

The Smithery CLI will:
1. Build the Docker container
2. Upload it to Smithery's infrastructure
3. Deploy the server and provide you with a URL

## Local Testing

To test the Smithery configuration locally:

```bash
# Build the Docker image
docker build -t nexonco-mcp .

# Run the container with default Smithery port
docker run -p 8081:8081 nexonco-mcp

# Or specify a custom port
docker run -e PORT=3000 -p 3000:3000 nexonco-mcp
```

Test the endpoints:
- Homepage: http://localhost:8081
- Health check: http://localhost:8081/health
- MCP endpoint: http://localhost:8081/mcp
- SSE endpoint: http://localhost:8081/sse

## Important Notes

### Custom Container Deprecation

⚠️ **Note**: Smithery is phasing out custom container hosting in favor of TypeScript servers for better performance on their edge infrastructure. While this Python-based deployment will continue to work, consider migrating to TypeScript for long-term support.

### Environment Variables

The server respects the following environment variables:
- `PORT` - Server port (default: 8081)

### CORS Configuration

The server is configured with permissive CORS settings for Smithery compatibility:
- `Access-Control-Allow-Origin: *`
- `Access-Control-Allow-Credentials: true`
- All methods and headers are allowed

For production deployments, you may want to restrict these settings based on your security requirements.

## Troubleshooting

### Build Fails

If the Docker build fails:
1. Ensure all dependencies are listed in `pyproject.toml`
2. Check that the Python version matches (>=3.11)
3. Verify uv is properly installed in the base image

### Server Won't Start

If the server won't start on Smithery:
1. Check the Smithery deployment logs
2. Verify the PORT environment variable is set correctly
3. Ensure all required dependencies are installed

### Connection Issues

If you can't connect to the deployed server:
1. Verify the `/mcp` endpoint is accessible
2. Check CORS headers in the response
3. Ensure the server is listening on 0.0.0.0 (not localhost)

## Additional Resources

- [Smithery Documentation](https://smithery.ai/docs)
- [MCP Protocol Specification](https://modelcontextprotocol.io)
- [Nexonco GitHub Repository](https://github.com/Nexgene-Research/nexonco-mcp)

## Support

For issues specific to:
- **Smithery deployment**: Contact Smithery support or check their documentation
- **Nexonco server**: Open an issue on [GitHub](https://github.com/Nexgene-Research/nexonco-mcp/issues)
