#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Local Pipecat Bot Demo with WebRTC UI

A standalone server that runs the Pipecat bot with a web UI for local testing.
No WhatsApp required - just open http://localhost:7860 in your browser.

Usage:
    uv run bot-local.py
    # or with custom host/port
    uv run bot-local.py --host 0.0.0.0 --port 8080
"""

import argparse
import asyncio
import os
import sys
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

load_dotenv(override=True)

# System instruction for the bot
SYSTEM_INSTRUCTION = """
You are Sudarshan Chatbot, a friendly, helpful and smart AI.

Your goal is to demonstrate your capabilities in a succinct way.

Your output will be converted to audio so don't include special characters in your answers.

Respond to what the user said in a creative and helpful way. Keep your responses brief. One or two sentences at most.
"""

# HTML UI for the bot
HTML_UI = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pipecat Voice Bot - Local Demo</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }

        .container {
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            max-width: 500px;
            width: 100%;
        }

        h1 {
            color: #333;
            margin-bottom: 10px;
            font-size: 28px;
            text-align: center;
        }

        .subtitle {
            color: #666;
            text-align: center;
            margin-bottom: 30px;
            font-size: 14px;
        }

        .status {
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            font-weight: 500;
            text-align: center;
            transition: all 0.3s ease;
        }

        .status.disconnected {
            background: #fee;
            color: #c33;
        }

        .status.connecting {
            background: #ffeaa7;
            color: #d63031;
        }

        .status.connected {
            background: #d4edda;
            color: #155724;
        }

        button {
            width: 100%;
            padding: 15px;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        #connectBtn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }

        #connectBtn:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
        }

        #disconnectBtn {
            background: #dc3545;
            color: white;
            margin-top: 10px;
        }

        #disconnectBtn:hover:not(:disabled) {
            background: #c82333;
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(220, 53, 69, 0.3);
        }

        .info {
            margin-top: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
            font-size: 14px;
            color: #666;
        }

        .info h3 {
            color: #333;
            margin-bottom: 10px;
            font-size: 16px;
        }

        .info ul {
            margin-left: 20px;
            margin-top: 10px;
        }

        .info li {
            margin: 5px 0;
        }

        .spinner {
            display: inline-block;
            width: 14px;
            height: 14px;
            border: 2px solid #f3f3f3;
            border-top: 2px solid #d63031;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-right: 8px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎙️ Pipecat Voice Bot</h1>
        <p class="subtitle">Local WebRTC Demo</p>

        <div id="status" class="status disconnected">
            Disconnected
        </div>

        <button id="connectBtn">Connect to Bot</button>
        <button id="disconnectBtn" style="display: none;">Disconnect</button>

        <div class="info">
            <h3>📝 Instructions</h3>
            <ul>
                <li>Click "Connect to Bot" to start</li>
                <li>Allow microphone access when prompted</li>
                <li>Start speaking - the bot will respond!</li>
                <li>Click "Disconnect" when done</li>
            </ul>
        </div>
    </div>

    <audio id="remoteAudio" autoplay></audio>

    <script>
        let peerConnection = null;
        let localStream = null;
        let remoteStream = null;

        const connectBtn = document.getElementById('connectBtn');
        const disconnectBtn = document.getElementById('disconnectBtn');
        const statusDiv = document.getElementById('status');
        const remoteAudio = document.getElementById('remoteAudio');

        function updateStatus(status, message) {
            statusDiv.className = `status ${status}`;
            statusDiv.innerHTML = message;
        }

        connectBtn.addEventListener('click', async () => {
            try {
                connectBtn.disabled = true;
                updateStatus('connecting', '<span class="spinner"></span>Connecting...');

                // Get user media
                localStream = await navigator.mediaDevices.getUserMedia({
                    audio: true,
                    video: false
                });

                // Create peer connection
                peerConnection = new RTCPeerConnection({
                    iceServers: [
                        { urls: 'stun:stun.l.google.com:19302' },
                        { urls: 'stun:stun1.l.google.com:19302' }
                    ]
                });

                // Add local tracks
                localStream.getTracks().forEach(track => {
                    peerConnection.addTrack(track, localStream);
                });

                // Handle remote tracks
                remoteStream = new MediaStream();
                peerConnection.ontrack = (event) => {
                    event.streams[0].getTracks().forEach(track => {
                        remoteStream.addTrack(track);
                    });
                    remoteAudio.srcObject = remoteStream;
                };

                // Handle connection state changes
                peerConnection.onconnectionstatechange = () => {
                    console.log('Connection state:', peerConnection.connectionState);
                    if (peerConnection.connectionState === 'connected') {
                        updateStatus('connected', '✓ Connected - Start speaking!');
                        connectBtn.style.display = 'none';
                        disconnectBtn.style.display = 'block';
                    } else if (peerConnection.connectionState === 'disconnected' ||
                               peerConnection.connectionState === 'failed') {
                        disconnect();
                    }
                };

                // Create offer
                const offer = await peerConnection.createOffer({
                    offerToReceiveAudio: true,
                    offerToReceiveVideo: false
                });
                await peerConnection.setLocalDescription(offer);

                // Send offer to server
                const response = await fetch('/api/offer', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        sdp: offer.sdp,
                        type: offer.type
                    })
                });

                if (!response.ok) {
                    throw new Error('Failed to connect to server');
                }

                const answer = await response.json();
                await peerConnection.setRemoteDescription(new RTCSessionDescription(answer));

            } catch (error) {
                console.error('Connection error:', error);
                updateStatus('disconnected', '✗ Connection failed: ' + error.message);
                disconnect();
            }
        });

        disconnectBtn.addEventListener('click', disconnect);

        function disconnect() {
            if (peerConnection) {
                peerConnection.close();
                peerConnection = null;
            }

            if (localStream) {
                localStream.getTracks().forEach(track => track.stop());
                localStream = null;
            }

            if (remoteStream) {
                remoteStream.getTracks().forEach(track => track.stop());
                remoteStream = null;
            }

            updateStatus('disconnected', 'Disconnected');
            connectBtn.style.display = 'block';
            connectBtn.disabled = false;
            disconnectBtn.style.display = 'none';
        }
    </script>
</body>
</html>
"""


async def run_bot(webrtc_connection):
    """Run the Pipecat bot with the given WebRTC connection"""
    try:
        pipecat_transport = SmallWebRTCTransport(
            webrtc_connection=webrtc_connection,
            params=TransportParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
                audio_out_10ms_chunks=2,
            ),
        )

        llm = GeminiLiveLLMService(
            model="models/gemini-2.5-flash-native-audio-preview-09-2025",
            api_key=os.getenv("GOOGLE_API_KEY"),
            voice_id="Kore",  # Aoede, Charon, Fenrir, Kore, Puck
            system_instruction=SYSTEM_INSTRUCTION,
        )

        context = OpenAILLMContext(
            [
                {
                    "role": "user",
                    "content": "Start by greeting the user warmly in Hindi and introducing yourself.",
                }
            ],
        )
        context_aggregator = llm.create_context_aggregator(context)

        pipeline = Pipeline(
            [
                pipecat_transport.input(),
                context_aggregator.user(),
                llm,
                pipecat_transport.output(),
                context_aggregator.assistant(),
            ]
        )

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
        )

        @pipecat_transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            logger.info("Client connected to bot")
            await task.queue_frames([LLMRunFrame()])

        @pipecat_transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info("Client disconnected from bot")
            await task.cancel()

        runner = PipelineRunner(handle_sigint=False)
        await runner.run(task)

    except Exception as e:
        logger.error(f"Error running bot: {e}")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    logger.info("Starting Pipecat local demo server")
    yield
    logger.info("Shutting down Pipecat local demo server")


# Create FastAPI app
app = FastAPI(
    title="Pipecat Local Demo",
    description="Local WebRTC voice bot demo",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the HTML UI"""
    return HTML_UI


@app.post("/api/offer")
async def handle_offer(request: Request, background_tasks: BackgroundTasks):
    """Handle WebRTC offer from client"""
    try:
        body = await request.json()
        sdp = body.get("sdp")
        sdp_type = body.get("type")

        if not sdp or not sdp_type:
            return JSONResponse(
                status_code=400,
                content={"error": "Missing SDP or type in request"}
            )

        logger.debug("Received WebRTC offer from client")

        # Create WebRTC connection with ICE servers
        webrtc_connection = SmallWebRTCConnection(
            ice_servers=[
                "stun:stun.l.google.com:19302",
                "stun:stun1.l.google.com:19302"
            ]
        )

        # Initialize the connection with the client's offer
        await webrtc_connection.initialize(sdp=sdp, type=sdp_type)

        # Connect the peer connection
        await webrtc_connection.connect()

        # Get the answer to send back to client
        answer = webrtc_connection.get_answer()

        # Start the bot in background
        background_tasks.add_task(run_bot, webrtc_connection)

        logger.info(f"WebRTC connection established (pc_id: {answer.get('pc_id')})")

        return JSONResponse(content=answer)

    except Exception as e:
        logger.error(f"Error handling WebRTC offer: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipecat Local Demo - Voice bot with web UI"
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Host to bind server to (default: localhost)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to bind server to (default: 7860)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    args = parser.parse_args()

    # Configure logging
    logger.remove(0)
    log_level = "DEBUG" if args.verbose else "INFO"
    logger.add(sys.stderr, level=log_level)

    # Check for required environment variables
    if not os.getenv("GOOGLE_API_KEY"):
        logger.error("GOOGLE_API_KEY environment variable is required!")
        logger.info("Please set it in your .env file or export it:")
        logger.info("  export GOOGLE_API_KEY='your-api-key-here'")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("🎙️  Pipecat Local Demo Server")
    logger.info("=" * 60)
    logger.info(f"Server starting on http://{args.host}:{args.port}")
    logger.info(f"Open your browser and navigate to: http://{args.host}:{args.port}")
    logger.info("=" * 60)

    # Run the server
    try:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_config=None,  # We're using loguru
        )
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)
