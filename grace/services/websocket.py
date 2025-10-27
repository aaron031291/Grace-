# -*- coding: utf-8 -*-
import asyncio
import json
import logging
import websockets
from websockets.server import WebSocketServerProtocol, WebSocketServer
from typing import Set

logger = logging.getLogger(__name__)

class WebSocketService:
    """
    Manages WebSocket communication for real-time updates between the frontend and backend.
    """

    def __init__(self, service_registry, host: str = "0.0.0.0", port: int = 8765):
        self.service_registry = service_registry
        self.host = host
        self.port = port
        self.server: WebSocketServer | None = None
        self.clients: Set[WebSocketServerProtocol] = set()
        logger.info(f"WebSocketService configured for ws://{self.host}:{self.port}")

    async def start(self):
        """Starts the WebSocket server as a long-running task."""
        if self.server:
            logger.warning("WebSocket server is already running.")
            return
        try:
            self.server = await websockets.serve(
                self._connection_handler, self.host, self.port
            )
            logger.info(f"✓ WebSocket server started and listening on ws://{self.host}:{self.port}")
        except Exception as e:
            logger.error(f"✗ Failed to start WebSocket server: {e}", exc_info=True)
            raise

    async def shutdown(self):
        """Stops the WebSocket server and closes all connections."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            logger.info("WebSocket server shut down.")
            self.server = None
        
        # Close all client connections
        for client in list(self.clients):
            await client.close(reason="Server shutting down")
        self.clients.clear()
        logger.info("All WebSocket client connections closed.")

    async def _connection_handler(self, websocket: WebSocketServerProtocol):
        """Handles a new client connection and its lifecycle."""
        self.clients.add(websocket)
        logger.info(f"New client connected: {websocket.remote_address}. Total clients: {len(self.clients)}")
        try:
            async for message in websocket:
                try:
                    # For now, just log and echo the message.
                    # In a real implementation, this would parse the message
                    # and dispatch actions to other services via the service_registry.
                    data = json.loads(message)
                    logger.debug(f"Received message from {websocket.remote_address}: {data}")
                    
                    # Example of dispatching to another service could go here.
                    # e.g., self.service_registry.get('task_manager').add_task(...)

                    # Echo the message back to the sender
                    await websocket.send(json.dumps({"status": "received", "data": data}))

                except json.JSONDecodeError:
                    logger.warning(f"Received invalid JSON from {websocket.remote_address}: {message}")
                    await websocket.send(json.dumps({"error": "Invalid JSON format"}))
                except Exception as e:
                    logger.error(f"Error processing message from {websocket.remote_address}: {e}", exc_info=True)
                    await websocket.send(json.dumps({"error": "An internal error occurred"}))
        except websockets.exceptions.ConnectionClosed as e:
            logger.info(f"Client {websocket.remote_address} disconnected: {e.reason} (code: {e.code})")
        finally:
            self.clients.remove(websocket)
            logger.info(f"Client removed: {websocket.remote_address}. Total clients: {len(self.clients)}")

    async def broadcast(self, message: dict):
        """Sends a message to all connected clients."""
        if not self.clients:
            return
        
        logger.debug(f"Broadcasting message to {len(self.clients)} clients: {message}")
        payload = json.dumps(message)
        # Create a list of tasks to send messages concurrently
        tasks = [client.send(payload) for client in self.clients]
        await asyncio.gather(*tasks, return_exceptions=True)
