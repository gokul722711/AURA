"use client";

import { useEffect, useRef } from "react";

/**
 * SphericalNodeField
 *
 * Renders an elegant 3D spherical field of orbiting nodes on an HTML5 canvas.
 * Conveys information being gathered and organized into structured evidence.
 * Respects `prefers-reduced-motion` for accessibility and avoids heavy 3D frameworks.
 */
export default function SphericalNodeField({
  size = 200,
  nodeCount = 130,
  className = "",
}) {
  const canvasRef = useRef(null);
  const animFrameRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Detect reduced motion preference
    const prefersReducedMotion =
      typeof window !== "undefined" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    // High-DPI scaling
    const dpr = typeof window !== "undefined" ? window.devicePixelRatio || 1 : 1;
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    canvas.style.width = `${size}px`;
    canvas.style.height = `${size}px`;
    ctx.scale(dpr, dpr);

    const radius = size * 0.38;
    const centerX = size / 2;
    const centerY = size / 2;
    const fov = 300;

    // Generate points uniformly distributed on a sphere (Fibonacci sphere)
    const nodes = [];
    const phi = Math.PI * (3 - Math.sqrt(5)); // Golden angle

    for (let i = 0; i < nodeCount; i++) {
      const y = 1 - (i / (nodeCount - 1)) * 2; // y goes from 1 to -1
      const radiusAtY = Math.sqrt(1 - y * y);
      const theta = phi * i;

      const x = Math.cos(theta) * radiusAtY;
      const z = Math.sin(theta) * radiusAtY;

      nodes.push({
        x: x * radius,
        y: y * radius,
        z: z * radius,
        baseX: x * radius,
        baseY: y * radius,
        baseZ: z * radius,
        phase: Math.random() * Math.PI * 2,
      });
    }

    let rotY = 0;
    let rotX = 0.25;
    let time = 0;

    const render = () => {
      ctx.clearRect(0, 0, size, size);

      if (!prefersReducedMotion) {
        rotY += 0.007;
        rotX += 0.0015;
        time += 0.02;
      }

      const cosY = Math.cos(rotY);
      const sinY = Math.sin(rotY);
      const cosX = Math.cos(rotX);
      const sinX = Math.sin(rotX);

      // Transform nodes
      const projected = [];
      for (let i = 0; i < nodes.length; i++) {
        const node = nodes[i];

        // Subtle organic breathing oscillation
        const breath = prefersReducedMotion ? 1 : 1 + Math.sin(time + node.phase) * 0.035;
        const curX = node.baseX * breath;
        const curY = node.baseY * breath;
        const curZ = node.baseZ * breath;

        // Rotation around Y axis
        const x1 = curX * cosY - curZ * sinY;
        const z1 = curZ * cosY + curX * sinY;

        // Rotation around X axis
        const y2 = curY * cosX - z1 * sinX;
        const z2 = z1 * cosX + curY * sinX;

        // Perspective projection
        const scale = fov / (fov + z2);
        const px = centerX + x1 * scale;
        const py = centerY + y2 * scale;

        projected.push({
          x: px,
          y: py,
          z: z2,
          scale,
        });
      }

      // Draw subtle connecting lines between nearby front nodes
      ctx.lineWidth = 0.75;
      const maxDist = radius * 0.42;

      for (let i = 0; i < projected.length; i++) {
        const p1 = projected[i];
        if (p1.z > radius * 0.2) continue; // Only connect nodes towards front/center

        for (let j = i + 1; j < projected.length; j++) {
          const p2 = projected[j];
          if (p2.z > radius * 0.2) continue;

          const dx = p1.x - p2.x;
          const dy = p1.y - p2.y;
          const dist = Math.sqrt(dx * dx + dy * dy);

          if (dist < maxDist) {
            const alpha = (1 - dist / maxDist) * 0.18 * Math.min(p1.scale, p2.scale);
            ctx.strokeStyle = `rgba(200, 215, 240, ${alpha.toFixed(3)})`;
            ctx.beginPath();
            ctx.moveTo(p1.x, p1.y);
            ctx.lineTo(p2.x, p2.y);
            ctx.stroke();
          }
        }
      }

      // Sort by z-order so front dots draw on top
      projected.sort((a, b) => b.z - a.z);

      // Draw nodes
      for (let i = 0; i < projected.length; i++) {
        const p = projected[i];

        // Depth-based sizing & opacity
        // z ranges from -radius to +radius
        // closer (negative z) -> larger & brighter
        const depthNorm = (-p.z + radius) / (2 * radius); // 0 (far) to 1 (close)
        const dotRadius = Math.max(0.8, (0.9 + depthNorm * 1.6) * (p.scale * 0.95));
        const opacity = Math.min(1, Math.max(0.18, 0.2 + depthNorm * 0.75));

        ctx.fillStyle = `rgba(240, 245, 255, ${opacity.toFixed(3)})`;
        ctx.beginPath();
        ctx.arc(p.x, p.y, dotRadius, 0, Math.PI * 2);
        ctx.fill();

        // Very faint subtle radiance for closest nodes
        if (depthNorm > 0.8 && !prefersReducedMotion) {
          ctx.fillStyle = `rgba(255, 255, 255, ${(opacity * 0.25).toFixed(3)})`;
          ctx.beginPath();
          ctx.arc(p.x, p.y, dotRadius * 2.2, 0, Math.PI * 2);
          ctx.fill();
        }
      }

      if (!prefersReducedMotion) {
        animFrameRef.current = requestAnimationFrame(render);
      }
    };

    render();

    return () => {
      if (animFrameRef.current) {
        cancelAnimationFrame(animFrameRef.current);
      }
    };
  }, [size, nodeCount]);

  return (
    <div
      className={`spherical-node-field ${className}`}
      style={{
        width: size,
        height: size,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
      }}
      aria-hidden="true"
    >
      <canvas ref={canvasRef} />
    </div>
  );
}
