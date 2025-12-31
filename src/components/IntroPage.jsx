import React, { useEffect, useRef, useState } from 'react';
import './IntroPage.css';

export function IntroPage({ onEnter }) {
    const canvasRef = useRef(null);
    const [showVideo, setShowVideo] = useState(false);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;

        const chars = '01ABCDEF';
        const fontSize = 14;
        const columns = canvas.width / fontSize;
        const drops = Array(Math.floor(columns)).fill(1);

        const draw = () => {
            ctx.fillStyle = 'rgba(15, 20, 25, 0.05)';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            ctx.fillStyle = '#0f0'; // Hacker green
            ctx.font = `${fontSize}px monospace`;

            for (let i = 0; i < drops.length; i++) {
                const text = chars[Math.floor(Math.random() * chars.length)];
                ctx.fillText(text, i * fontSize, drops[i] * fontSize);

                if (drops[i] * fontSize > canvas.height && Math.random() > 0.975) {
                    drops[i] = 0;
                }
                drops[i]++;
            }
        };

        const interval = setInterval(draw, 33);

        const handleResize = () => {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
        };

        window.addEventListener('resize', handleResize);

        return () => {
            clearInterval(interval);
            window.removeEventListener('resize', handleResize);
        };
    }, []);

    return (
        <div className="intro-container">
            {/* Background Matrix Effect */}
            <canvas ref={canvasRef} className="matrix-canvas" />

            <div className="intro-content">
                <h1 className="glitch-text" data-text="IDS ENSEMBLE">IDS ENSEMBLE</h1>
                <p className="subtitle">Advanced Intrusion Detection System</p>

                <div className="simulation-status">
                    <span className="blink">●</span> SYSTEM INITIALIZED
                </div>

                <div className="button-group">
                    <button className="watch-button" onClick={() => setShowVideo(true)}>
                        <span>▶</span> Watch Simulation
                    </button>

                    <button className="enter-button" onClick={onEnter}>
                        INITIALIZE HOST
                        <span className="arrow">→</span>
                    </button>
                </div>
            </div>

            {/* Video Modal */}
            {showVideo && (
                <div className="video-modal-overlay" onClick={() => setShowVideo(false)}>
                    <div className="video-modal-content" onClick={e => e.stopPropagation()}>
                        <button className="close-modal" onClick={() => setShowVideo(false)}>×</button>
                        <video
                            controls
                            autoPlay
                            className="modal-video"
                        >
                            <source src="/ids_animation.mp4" type="video/mp4" />
                            Your browser does not support the video tag.
                        </video>
                    </div>
                </div>
            )}
        </div>
    );
}
