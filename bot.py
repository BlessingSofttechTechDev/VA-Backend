#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
import os
import asyncio
from contextlib import suppress

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    Frame,
    LLMMessagesAppendFrame,
    LLMRunFrame,
    TranscriptionFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.google.stt import GoogleSTTService
from pipecat.services.google.tts import GoogleHttpTTSService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

from hr_rag import retrieve_hr_policies_context


class RAGContextInjector(FrameProcessor):
    """Intercepts user transcriptions and injects RAG-retrieved HR policy context.

    Sits between STT and the LLM context aggregator. When a TranscriptionFrame
    arrives, it:
      1. Calls the RAG retriever with the user's text.
      2. Pushes an LLMMessagesAppendFrame containing the policy excerpts as a
         system message so the LLM sees them before generating a response.
      3. Passes the original TranscriptionFrame downstream unchanged.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._pending_text: list[str] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Collect transcription text
        if isinstance(frame, TranscriptionFrame):
            user_text = frame.text.strip()
            if user_text:
                self._pending_text.append(user_text)
                logger.debug(f"RAGContextInjector: captured transcription '{user_text}'")

        # When the LLM is about to run, inject RAG context first
        if isinstance(frame, LLMRunFrame) and self._pending_text:
            query = " ".join(self._pending_text)
            self._pending_text.clear()
            logger.debug(f"RAGContextInjector: running retrieval for query '{query}'")

            snippets = retrieve_hr_policies_context(query)
            if snippets:
                context_block = "\n\n".join(snippets)
                rag_message = {
                    "role": "system",
                    "content": f"HR policy excerpts relevant to the user's question:\n\n{context_block}",
                }
                logger.debug(
                    f"RAGContextInjector: injecting {len(snippets)} policy snippet(s) into context"
                )
                # Push the RAG context message before the LLMRunFrame
                await self.push_frame(LLMMessagesAppendFrame(messages=[rag_message]), direction)
            else:
                logger.debug("RAGContextInjector: no relevant policy snippets found")

        # Always pass the frame downstream
        await self.push_frame(frame, direction)


load_dotenv(override=True)

SYSTEM_INSTRUCTION = """
You are an HR assistant chatbot for SPIL (Sirca Paints India Ltd).

Use the HR policy excerpts provided to you to answer the user's questions.
You may summarize, explain, or describe what is covered in the excerpts.
Do not invent policies that are not in the excerpts.

If the user asks a specific question and the answer is clearly not present in the
excerpts, say: "This is not mentioned in the HR policies document."

When possible, briefly mention the page number from which you took the information.
Keep your responses concise: one or two sentences at most, suitable for text-to-speech.
"""


async def run_bot(webrtc_connection):
    logger.debug(
        f"run_bot() starting for WebRTC connection: {getattr(webrtc_connection, 'pc_id', webrtc_connection)}"
    )

    pipecat_transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            audio_out_10ms_chunks=2,
        ),
    )
    logger.debug(
        "SmallWebRTCTransport initialized with audio_in_enabled=True, "
        "audio_out_enabled=True, vad=SileroVADAnalyzer, audio_out_10ms_chunks=2"
    )

    # Use a dedicated HTTP session for Google STT/TTS services for this call.
    async with aiohttp.ClientSession() as session:
        # STT: audio -> text
        stt = GoogleSTTService(
            session=session,
            credentials_path=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
            language="en-US",
        )
        logger.debug(
            "GoogleSTTService initialized with language='en-US' and "
            f"credentials_path={os.getenv('GOOGLE_APPLICATION_CREDENTIALS')!r}"
        )

        # TTS: text -> audio
        tts = GoogleHttpTTSService(
            session=session,
            credentials_path=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
            voice_id="en-US-Wavenet-D",
            sample_rate=24000,
        )
        logger.debug(
            "GoogleHttpTTSService initialized with voice_id='en-US-Wavenet-D', "
            "sample_rate=24000"
        )

        # Text LLM
        llm = GoogleLLMService(
            api_key=os.getenv("GOOGLE_API_KEY"),
            voice_id="Puck",
            system_instruction=SYSTEM_INSTRUCTION,
        )
        logger.debug(
            "GoogleLLMService initialized with voice_id='Puck'. "
            f"GOOGLE_API_KEY present={bool(os.getenv('GOOGLE_API_KEY'))}"
        )

        initial_messages = [
            {
                "role": "system",
                "content": (
                    "You will answer questions about the company's HR policies. "
                    "Use only the HR policy excerpts that may be provided to you."
                ),
            },
            {
                "role": "user",
                "content": "Start by greeting the user warmly and introducing yourself.",
            },
        ]

        context = LLMContext(initial_messages)
        context_aggregator = LLMContextAggregatorPair(context)
        logger.debug(
            "LLMContext and LLMContextAggregatorPair initialized with initial user prompt "
            "'Start by greeting the user warmly and introducing yourself.'"
        )

        # RAG processor: intercepts user text, retrieves HR policy context, injects it
        rag_injector = RAGContextInjector()
        logger.debug("RAGContextInjector created to inject HR policy context before LLM")

        pipeline = Pipeline(
            [
                pipecat_transport.input(),   # audio from WhatsApp
                stt,                         # STT: audio -> text
                rag_injector,                # RAG: retrieve & inject policy context
                context_aggregator.user(),   # update user context
                llm,                         # Google LLM
                tts,                         # TTS: text -> audio
                pipecat_transport.output(),  # audio back to WhatsApp
                context_aggregator.assistant(),  # update assistant context
            ]
        )
        logger.debug(
            "Pipeline constructed: SmallWebRTCInputTransport -> GoogleSTTService -> "
            "RAGContextInjector -> LLMUserAggregator -> GoogleLLMService -> "
            "GoogleHttpTTSService -> SmallWebRTCOutputTransport -> LLMAssistantAggregator"
        )

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
        )
        logger.debug("PipelineTask created with metrics and usage_metrics enabled")

        @pipecat_transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            logger.info("Pipecat Client connected")
            # Kick off the conversation.
            logger.debug(
                "on_client_connected: queuing initial LLMRunFrame() to trigger greeting response"
            )
            await task.queue_frames([LLMRunFrame()])

        @pipecat_transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info("Pipecat Client disconnected")
            await task.cancel()
            logger.debug("PipelineTask cancel() requested from on_client_disconnected")

        runner = PipelineRunner(handle_sigint=False)

        async def log_context_changes() -> None:
            """Periodically log the full LLM context so we can see STT transcripts and LLM replies."""
            messages = getattr(context, "messages", [])
            prev_len = len(messages)
            logger.debug(f"Initial LLM context messages ({prev_len}): {messages}")
            try:
                while True:
                    await asyncio.sleep(2.0)
                    messages = getattr(context, "messages", [])
                    if len(messages) != prev_len:
                        logger.debug(
                            f"Updated LLM context messages ({len(messages)}): {messages}"
                        )
                        prev_len = len(messages)
            except asyncio.CancelledError:
                logger.debug("Context logging task cancelled")

        logger.debug(
            f"PipelineRunner starting for connection: {getattr(webrtc_connection, 'pc_id', webrtc_connection)}"
        )
        context_log_task = asyncio.create_task(log_context_changes())
        try:
            await runner.run(task)
        finally:
            context_log_task.cancel()
            with suppress(asyncio.CancelledError):
                await context_log_task

        logger.debug(
            f"PipelineRunner finished for connection: {getattr(webrtc_connection, 'pc_id', webrtc_connection)}"
        )

