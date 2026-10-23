import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  UploadCloud,
  Crosshair,
  Sliders,
  RotateCcw,
  AlertCircle,
  HelpCircle,
  CheckCircle2,
  AlertTriangle,
  Flame
} from 'lucide-react';
import { API_BASE_URL } from '../config';
import type { PredictResponse, ClassName } from '../types';

interface AnalyzeViewProps {
  backendReady: boolean;
  device?: string;
}

type ViewerMode = 'original' | 'heatmap' | 'slider';

export const AnalyzeView: React.FC<AnalyzeViewProps> = ({ backendReady, device = 'CUDA' }) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [imageDimensions, setImageDimensions] = useState<{ width: number; height: number } | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);

  // DICOM Viewer state
  const [viewerMode, setViewerMode] = useState<ViewerMode>('heatmap');
  const [overlayOpacity, setOverlayOpacity] = useState<number>(80);
  const [sliderPosition, setSliderPosition] = useState<number>(50);
  const [showReticle, setShowReticle] = useState<boolean>(false);
  const [selectedTargetClass, setSelectedTargetClass] = useState<ClassName | 'auto'>('auto');
  const [isExplaining, setIsExplaining] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const sliderContainerRef = useRef<HTMLDivElement>(null);
  const isDraggingSlider = useRef<boolean>(false);

  // Keyboard navigation shortcuts (O, H, S, Esc)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't trigger if user is in an input or select
      if (['INPUT', 'SELECT', 'TEXTAREA'].includes((e.target as HTMLElement)?.tagName)) {
        return;
      }
      if (e.key === 'o' || e.key === 'O') {
        setViewerMode('original');
      } else if (e.key === 'h' || e.key === 'H') {
        setViewerMode('heatmap');
      } else if (e.key === 's' || e.key === 'S') {
        setViewerMode('slider');
      } else if (e.key === 'Escape') {
        handleReset();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [selectedFile]);

  const handleReset = () => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setSelectedFile(null);
    setPreviewUrl(null);
    setImageDimensions(null);
    setResult(null);
    setError(null);
    setSelectedTargetClass('auto');
    setViewerMode('heatmap');
  };

  const validateAndLoadFile = (file: File) => {
    setError(null);

    // 1. Client-side mime type validation
    const validTypes = ['image/jpeg', 'image/png', 'image/jpg'];
    if (!validTypes.includes(file.type.toLowerCase())) {
      setError('Invalid file format. Please provide a standard knee radiograph (JPEG or PNG).');
      return;
    }

    // 2. Client-side size check (max 10 MB)
    if (file.size > 10 * 1024 * 1024) {
      setError(`File size exceeds 10 MB limit (${(file.size / (1024 * 1024)).toFixed(2)} MB).`);
      return;
    }

    const objectUrl = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      // 3. Client-side dimension check (max 4096 px)
      if (img.naturalWidth > 4096 || img.naturalHeight > 4096) {
        URL.revokeObjectURL(objectUrl);
        setError(`Image resolution (${img.naturalWidth}×${img.naturalHeight} px) exceeds maximum permitted 4096×4096 px.`);
        return;
      }

      setImageDimensions({ width: img.naturalWidth, height: img.naturalHeight });
      setSelectedFile(file);
      setPreviewUrl(objectUrl);
      setResult(null);
      setSelectedTargetClass('auto');

      // Automatically run analysis on drop/select for instant reading-room flow
      runInference(file);
    };
    img.onerror = () => {
      URL.revokeObjectURL(objectUrl);
      setError('Unable to parse image data. The file appears to be corrupted.');
    };
    img.src = objectUrl;
  };

  const runInference = async (fileToAnalyze: File, target?: ClassName) => {
    if (target) {
      setIsExplaining(true);
    } else {
      setLoading(true);
    }
    setError(null);

    const formData = new FormData();
    formData.append('file', fileToAnalyze);

    let url = `${API_BASE_URL}/predict`;
    if (target) {
      url += `?target_class=${encodeURIComponent(target)}`;
    }

    try {
      const response = await fetch(url, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server error: HTTP ${response.status} ${response.statusText}`);
      }

      const data: PredictResponse = await response.json();
      setResult(data);
      if (target) {
        setSelectedTargetClass(target);
      } else {
        setSelectedTargetClass('auto');
      }
    } catch (err: any) {
      console.error('Inference error:', err);
      setError(err.message || 'Failed to connect to inference server.');
    } finally {
      setLoading(false);
      setIsExplaining(false);
    }
  };

  const onDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const onDragLeave = () => {
    setIsDragging(false);
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      validateAndLoadFile(e.dataTransfer.files[0]);
    }
  };

  // Slider dragging logic for Split Slider
  const handleSliderMove = useCallback((clientX: number) => {
    if (!sliderContainerRef.current) return;
    const rect = sliderContainerRef.current.getBoundingClientRect();
    const x = clientX - rect.left;
    const percentage = Math.max(0, Math.min(100, (x / rect.width) * 100));
    setSliderPosition(percentage);
  }, []);

  const handleMouseDown = (e: React.MouseEvent) => {
    isDraggingSlider.current = true;
    handleSliderMove(e.clientX);
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDraggingSlider.current) return;
    handleSliderMove(e.clientX);
  };

  const handleMouseUp = () => {
    isDraggingSlider.current = false;
  };

  const handleTouchMove = (e: React.TouchEvent) => {
    if (e.touches[0]) {
      handleSliderMove(e.touches[0].clientX);
    }
  };

  // Color-blind safe class badges with clear clinical iconography
  const renderClassBadge = (className: ClassName) => {
    switch (className) {
      case 'normal':
        return (
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 font-mono text-sm tracking-wider uppercase font-semibold">
            <CheckCircle2 className="w-4 h-4 text-emerald-400" />
            <span>Normal</span>
          </div>
        );
      case 'osteopenia':
        return (
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg bg-amber-500/10 border border-amber-500/30 text-amber-400 font-mono text-sm tracking-wider uppercase font-semibold">
            <AlertTriangle className="w-4 h-4 text-amber-400" />
            <span>Osteopenia</span>
          </div>
        );
      case 'osteoporosis':
        return (
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg bg-rose-500/10 border border-rose-500/30 text-rose-400 font-mono text-sm tracking-wider uppercase font-semibold">
            <Flame className="w-4 h-4 text-rose-400" />
            <span>Osteoporosis</span>
          </div>
        );
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
      {/* Backend Offline Warning */}
      {!backendReady && (
        <div
          role="alert"
          className="p-4 rounded-xl bg-rose-950/40 border border-rose-800/80 text-rose-200 text-sm flex items-start gap-3"
        >
          <AlertCircle className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
          <div>
            <h2 className="font-semibold text-rose-100 mb-0.5">Backend Service Offline</h2>
            <p className="text-xs text-rose-300">
              The inference service is not reachable on <code className="font-mono">{API_BASE_URL}</code>. Launch the local backend with <code className="font-mono">run_app.bat</code> or start Uvicorn.
            </p>
          </div>
        </div>
      )}

      {/* Global Error Banner */}
      {error && (
        <div
          role="alert"
          className="p-3.5 rounded-xl bg-rose-950/30 border border-rose-800 text-rose-300 text-xs flex items-center justify-between gap-2"
        >
          <div className="flex items-center gap-2">
            <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
            <span>{error}</span>
          </div>
          <button
            onClick={() => setError(null)}
            className="text-xs text-rose-400 hover:text-rose-200 underline"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* STATE 1: Centered Drop Zone (Before Upload) */}
      {!selectedFile ? (
        <div className="py-12 sm:py-20 flex flex-col items-center justify-center">
          <div
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            onDrop={onDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`w-full max-w-2xl border-2 border-dashed rounded-2xl p-10 sm:p-16 flex flex-col items-center justify-center text-center cursor-pointer transition-all ${
              isDragging
                ? 'border-cyan-400 bg-cyan-950/20 shadow-[0_0_30px_rgba(0,229,255,0.15)]'
                : 'border-slate-800 hover:border-slate-700 bg-[#0d1219]/90 hover:bg-[#101721]'
            }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/jpeg,image/png,image/jpg"
              className="hidden"
              onChange={(e) => e.target.files?.[0] && validateAndLoadFile(e.target.files[0])}
            />

            <div className="w-16 h-16 rounded-2xl bg-slate-900 border border-slate-800 flex items-center justify-center text-cyan-400 mb-6 shadow-inner">
              <UploadCloud className="w-8 h-8" />
            </div>

            <h1 className="text-xl sm:text-2xl font-semibold text-slate-100 tracking-tight mb-2">
              Drop a knee radiograph
            </h1>

            <p className="text-sm text-slate-400 max-w-md mx-auto mb-6 leading-relaxed">
              Click to browse or drag a JPEG/PNG radiograph here. The model will classify bone density and project the Grad-CAM activation heatmap onto the film lightbox.
            </p>

            <div className="flex flex-wrap items-center justify-center gap-3 text-xs text-slate-500 font-mono">
              <span className="px-2.5 py-1 rounded-md bg-slate-900 border border-slate-800">JPEG / PNG</span>
              <span className="px-2.5 py-1 rounded-md bg-slate-900 border border-slate-800">Max 10 MB</span>
              <span className="px-2.5 py-1 rounded-md bg-slate-900 border border-slate-800">≤ 4096 px</span>
            </div>
          </div>
        </div>
      ) : (
        /* STATE 2: Reading Room Lightbox & Results (After Upload) */
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          {/* LEFT COLUMN: DICOM Lightbox Viewer (Wide) */}
          <div className="lg:col-span-8 space-y-4">
            {/* Lightbox Canvas Container */}
            <div className="rounded-2xl border border-slate-800 bg-[#000000] overflow-hidden shadow-2xl relative flex flex-col select-none">
              {/* Viewer Control Bar */}
              <div className="bg-[#0a0e14] border-b border-slate-800 px-4 py-2.5 flex flex-wrap items-center justify-between gap-3 text-xs">
                {/* Mode Segmented Buttons */}
                <div
                  role="tablist"
                  aria-label="Viewer Mode"
                  className="flex items-center p-0.5 rounded-lg bg-slate-900 border border-slate-800"
                >
                  <button
                    role="tab"
                    aria-selected={viewerMode === 'original'}
                    onClick={() => setViewerMode('original')}
                    className={`px-3 py-1 rounded-md font-medium transition-colors ${
                      viewerMode === 'original'
                        ? 'bg-slate-800 text-cyan-400'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                    title="Original Radiograph (Shortcut: O)"
                  >
                    Original <span className="opacity-50 text-[10px] font-mono">O</span>
                  </button>
                  <button
                    role="tab"
                    aria-selected={viewerMode === 'heatmap'}
                    onClick={() => setViewerMode('heatmap')}
                    className={`px-3 py-1 rounded-md font-medium transition-colors ${
                      viewerMode === 'heatmap'
                        ? 'bg-slate-800 text-cyan-400'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                    title="Grad-CAM Heatmap (Shortcut: H)"
                  >
                    Heatmap <span className="opacity-50 text-[10px] font-mono">H</span>
                  </button>
                  <button
                    role="tab"
                    aria-selected={viewerMode === 'slider'}
                    onClick={() => setViewerMode('slider')}
                    className={`px-3 py-1 rounded-md font-medium transition-colors ${
                      viewerMode === 'slider'
                        ? 'bg-slate-800 text-cyan-400'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                    title="Interactive Split Slider (Shortcut: S)"
                  >
                    Split <span className="opacity-50 text-[10px] font-mono">S</span>
                  </button>
                </div>

                {/* Right controls: Opacity slider (if heatmap) & Reticle toggle */}
                <div className="flex items-center gap-4">
                  {viewerMode === 'heatmap' && (
                    <div className="flex items-center gap-2">
                      <Sliders className="w-3.5 h-3.5 text-slate-500" />
                      <span className="text-slate-400">Blend:</span>
                      <input
                        type="range"
                        min="10"
                        max="100"
                        value={overlayOpacity}
                        onChange={(e) => setOverlayOpacity(Number(e.target.value))}
                        aria-label="Heatmap opacity"
                        className="w-20 sm:w-28 h-1 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-400"
                      />
                      <span className="font-mono tabular-nums text-slate-300 w-8 text-right">
                        {overlayOpacity}%
                      </span>
                    </div>
                  )}

                  {/* Crosshair / Reticle Grid Toggle */}
                  <button
                    onClick={() => setShowReticle((prev) => !prev)}
                    className={`p-1.5 rounded-md border text-xs flex items-center gap-1.5 transition-colors ${
                      showReticle
                        ? 'bg-cyan-950/40 border-cyan-500/50 text-cyan-400'
                        : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200'
                    }`}
                    title="Toggle Radiographic Grid Reticle"
                    aria-pressed={showReticle}
                  >
                    <Crosshair className="w-3.5 h-3.5" />
                    <span className="hidden sm:inline">Grid</span>
                  </button>
                </div>
              </div>

              {/* Lightbox Viewing Canvas */}
              <div
                ref={sliderContainerRef}
                onMouseDown={viewerMode === 'slider' ? handleMouseDown : undefined}
                onMouseMove={viewerMode === 'slider' ? handleMouseMove : undefined}
                onMouseUp={viewerMode === 'slider' ? handleMouseUp : undefined}
                onTouchMove={viewerMode === 'slider' ? handleTouchMove : undefined}
                className={`relative w-full aspect-square max-h-[620px] bg-black flex items-center justify-center overflow-hidden ${
                  viewerMode === 'slider' ? 'cursor-ew-resize' : ''
                }`}
              >
                {/* Optional Reticle Measurement Grid */}
                {showReticle && (
                  <div
                    aria-hidden="true"
                    className="absolute inset-0 dicom-reticle-grid pointer-events-none z-20"
                  />
                )}

                {/* Corner Tick Reticles (DICOM Aesthetic) */}
                <div aria-hidden="true" className="absolute top-3 left-3 w-4 h-4 border-t-2 border-l-2 border-cyan-400/40 pointer-events-none z-20" />
                <div aria-hidden="true" className="absolute top-3 right-3 w-4 h-4 border-t-2 border-r-2 border-cyan-400/40 pointer-events-none z-20" />
                <div aria-hidden="true" className="absolute bottom-3 left-3 w-4 h-4 border-b-2 border-l-2 border-cyan-400/40 pointer-events-none z-20" />
                <div aria-hidden="true" className="absolute bottom-3 right-3 w-4 h-4 border-b-2 border-r-2 border-cyan-400/40 pointer-events-none z-20" />

                {/* Monospace Metadata HUD Overlays */}
                {/* Top-Left: Filename & Native Resolution */}
                <div className="absolute top-4 left-5 z-20 pointer-events-none font-mono text-[11px] text-cyan-400/90 tracking-wider space-y-0.5 bg-black/60 px-2 py-1 rounded backdrop-blur-sm">
                  <div className="truncate max-w-[200px] text-slate-200">{selectedFile.name}</div>
                  <div className="tabular-nums text-slate-400">
                    {imageDimensions ? `${imageDimensions.width} × ${imageDimensions.height} px` : 'Loading...'}
                  </div>
                </div>

                {/* Top-Right: Latency & Device */}
                <div className="absolute top-4 right-5 z-20 pointer-events-none font-mono text-[11px] text-right space-y-0.5 bg-black/60 px-2 py-1 rounded backdrop-blur-sm">
                  <div className="tabular-nums text-cyan-400">
                    {result ? `${result.inference_ms} ms` : loading ? 'COMPUTING...' : 'IDLE'}
                  </div>
                  <div className="text-[10px] text-slate-400">{device}</div>
                </div>

                {/* Bottom-Left: Viewer Mode readout */}
                <div className="absolute bottom-4 left-5 z-20 pointer-events-none font-mono text-[11px] text-slate-400 bg-black/60 px-2 py-1 rounded backdrop-blur-sm">
                  {viewerMode === 'original' && 'MODE: ORIGINAL X-RAY'}
                  {viewerMode === 'heatmap' && 'MODE: GRAD-CAM HEATMAP (LAYER4)'}
                  {viewerMode === 'slider' && `SPLIT: ${Math.round(sliderPosition)}%`}
                </div>

                {/* Bottom-Right: Target Explanation status */}
                {result && (
                  <div className="absolute bottom-4 right-5 z-20 pointer-events-none font-mono text-[11px] text-slate-300 bg-black/60 px-2 py-1 rounded backdrop-blur-sm">
                    TARGET: <span className="text-cyan-400 uppercase">{selectedTargetClass === 'auto' ? result.predicted_class : selectedTargetClass}</span>
                  </div>
                )}

                {/* Scanning line animation during inference */}
                {loading && (
                  <div className="absolute inset-x-0 z-30 pointer-events-none animate-dicom-scan">
                    <div className="h-0.5 bg-cyan-400 shadow-[0_0_15px_#00e5ff]" />
                  </div>
                )}

                {/* Image Render Layer: Mode 1 - Original */}
                {viewerMode === 'original' && (
                  <img
                    src={previewUrl!}
                    alt="Original Knee Radiograph"
                    className="w-full h-full object-contain"
                  />
                )}

                {/* Image Render Layer: Mode 2 - Heatmap with smooth crossfade */}
                {viewerMode === 'heatmap' && (
                  <div className="relative w-full h-full">
                    {/* Base X-Ray */}
                    <img
                      src={previewUrl!}
                      alt="Original Knee Radiograph base"
                      className="absolute inset-0 w-full h-full object-contain"
                    />
                    {/* Grad-CAM Overlay */}
                    {result && (
                      <img
                        src={`data:image/png;base64,${result.gradcam_overlay_png_base64}`}
                        alt="Grad-CAM Jet Activation Heatmap"
                        className="absolute inset-0 w-full h-full object-contain transition-opacity duration-300"
                        style={{ opacity: overlayOpacity / 100 }}
                      />
                    )}
                  </div>
                )}

                {/* Image Render Layer: Mode 3 - Split Slider */}
                {viewerMode === 'slider' && result && (
                  <div className="relative w-full h-full">
                    {/* Background: Grad-CAM Overlay */}
                    <img
                      src={`data:image/png;base64,${result.gradcam_overlay_png_base64}`}
                      alt="Grad-CAM Overlay Background"
                      className="absolute inset-0 w-full h-full object-contain pointer-events-none"
                    />

                    {/* Foreground: Original Image clipped to sliderPosition */}
                    <div
                      className="absolute inset-0 overflow-hidden pointer-events-none"
                      style={{ width: `${sliderPosition}%` }}
                    >
                      <img
                        src={previewUrl!}
                        alt="Original Knee X-ray Foreground"
                        className="absolute inset-0 w-full h-full object-contain pointer-events-none"
                        style={{
                          maxWidth: 'none',
                          width: sliderContainerRef.current ? `${sliderContainerRef.current.clientWidth}px` : '100%',
                          height: '100%'
                        }}
                      />
                    </div>

                    {/* Thin vertical curtain divider */}
                    <div
                      className="absolute top-0 bottom-0 w-0.5 bg-cyan-400 shadow-[0_0_10px_#00e5ff] pointer-events-none z-20"
                      style={{ left: `${sliderPosition}%` }}
                    >
                      <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-6 h-6 rounded-full bg-black border-2 border-cyan-400 shadow-lg flex items-center justify-center text-cyan-400 text-[10px] font-mono">
                        ⇄
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Grad-CAM Color-Scale Legend Bar (Phosphor / Jet scale) */}
              {viewerMode !== 'original' && result && (
                <div className="bg-[#0a0e14] border-t border-slate-800 px-5 py-2.5 flex items-center justify-between text-[11px] font-mono text-slate-400">
                  <span>Grad-CAM Activation:</span>
                  <div className="flex items-center gap-2">
                    <span className="text-slate-500">0.0 (Low)</span>
                    <div
                      aria-hidden="true"
                      className="w-36 sm:w-48 h-2.5 rounded-sm border border-slate-700 overflow-hidden"
                      style={{
                        background: 'linear-gradient(to right, #000080, #0000ff, #00ffff, #ffff00, #ff0000, #800000)'
                      }}
                    />
                    <span className="text-cyan-400">1.0 (High)</span>
                  </div>
                </div>
              )}
            </div>

            {/* Mandatory Grad-CAM Clinical Caution Note */}
            <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800 text-slate-400 text-xs flex items-start gap-2.5 leading-relaxed">
              <HelpCircle className="w-4 h-4 text-slate-500 shrink-0 mt-0.5" />
              <p>
                <strong>Grad-CAM note:</strong> Heatmaps show which regions influenced the model's output. They are not proof of clinical reasoning.
              </p>
            </div>
          </div>

          {/* RIGHT COLUMN: Results & Inspection Panel (Narrow ~360px) */}
          <div className="lg:col-span-4 space-y-5">
            <div className="rounded-2xl border border-slate-800 bg-[#0d1219] p-6 space-y-6 shadow-xl">
              {/* Header & Reset Button */}
              <div className="flex items-center justify-between border-b border-slate-800 pb-4">
                <div>
                  <h2 className="text-sm font-semibold text-slate-100 tracking-tight">Classification Result</h2>
                  <p className="text-[11px] text-slate-400 mt-0.5">ResNet-50 forward pass</p>
                </div>
                <button
                  onClick={handleReset}
                  className="px-2.5 py-1.5 rounded-lg border border-slate-800 hover:border-slate-700 bg-slate-900 text-slate-400 hover:text-slate-200 text-xs font-medium flex items-center gap-1.5 transition-colors"
                  title="Upload another image (Shortcut: Esc)"
                >
                  <RotateCcw className="w-3 h-3" />
                  <span>New image</span>
                </button>
              </div>

              {/* Large Predicted Class Display */}
              {loading ? (
                <div className="py-6 flex flex-col items-center justify-center space-y-3">
                  <div className="w-6 h-6 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin" />
                  <span className="text-xs font-mono text-cyan-400 animate-pulse">Running inference...</span>
                </div>
              ) : result ? (
                <div className="space-y-4">
                  <div>
                    <span className="text-[11px] font-mono uppercase tracking-wider text-slate-500 block mb-1.5">
                      Predicted Class
                    </span>
                    {renderClassBadge(result.predicted_class)}
                  </div>

                  {/* Model Output Probabilities */}
                  <div className="space-y-2 pt-2 border-t border-slate-800/80">
                    <div className="flex items-baseline justify-between">
                      <span className="text-xs font-semibold text-slate-200">
                        Model output probabilities
                      </span>
                      <span className="text-[11px] text-slate-500 italic">
                        Softmax outputs; not calibrated.
                      </span>
                    </div>

                    <div className="space-y-3 pt-1">
                      {/* Normal Class Bar */}
                      <div className="space-y-1">
                        <div className="flex justify-between text-xs font-mono">
                          <span className="text-slate-300">Normal</span>
                          <span className="tabular-nums text-slate-100 font-semibold">
                            {(result.probabilities.normal * 100).toFixed(2)}%
                          </span>
                        </div>
                        <div className="w-full h-1.5 rounded-full bg-slate-800 overflow-hidden">
                          <div
                            className="h-full bg-emerald-400 transition-all duration-700 ease-out rounded-full"
                            style={{ width: `${result.probabilities.normal * 100}%` }}
                          />
                        </div>
                      </div>

                      {/* Osteopenia Class Bar */}
                      <div className="space-y-1">
                        <div className="flex justify-between text-xs font-mono">
                          <span className="text-slate-300">Osteopenia</span>
                          <span className="tabular-nums text-slate-100 font-semibold">
                            {(result.probabilities.osteopenia * 100).toFixed(2)}%
                          </span>
                        </div>
                        <div className="w-full h-1.5 rounded-full bg-slate-800 overflow-hidden">
                          <div
                            className="h-full bg-amber-400 transition-all duration-700 ease-out rounded-full"
                            style={{ width: `${result.probabilities.osteopenia * 100}%` }}
                          />
                        </div>
                      </div>

                      {/* Osteoporosis Class Bar */}
                      <div className="space-y-1">
                        <div className="flex justify-between text-xs font-mono">
                          <span className="text-slate-300">Osteoporosis</span>
                          <span className="tabular-nums text-slate-100 font-semibold">
                            {(result.probabilities.osteoporosis * 100).toFixed(2)}%
                          </span>
                        </div>
                        <div className="w-full h-1.5 rounded-full bg-slate-800 overflow-hidden">
                          <div
                            className="h-full bg-rose-400 transition-all duration-700 ease-out rounded-full"
                            style={{ width: `${result.probabilities.osteoporosis * 100}%` }}
                          />
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Explain Class Segmented Control */}
                  <div className="space-y-2 pt-3 border-t border-slate-800/80">
                    <div className="flex items-center justify-between">
                      <label htmlFor="explain-class-segmented" className="text-xs font-semibold text-slate-200">
                        Explain class:
                      </label>
                      {isExplaining && (
                        <span className="text-[11px] font-mono text-cyan-400 animate-pulse">
                          Recomputing...
                        </span>
                      )}
                    </div>

                    <div
                      id="explain-class-segmented"
                      role="radiogroup"
                      aria-label="Explain class selector"
                      className="grid grid-cols-2 gap-1.5 p-1 rounded-xl bg-slate-900 border border-slate-800 text-xs"
                    >
                      <button
                        role="radio"
                        aria-checked={selectedTargetClass === 'auto'}
                        onClick={() => runInference(selectedFile)}
                        disabled={isExplaining}
                        className={`py-1.5 px-2 rounded-lg font-medium text-center transition-colors truncate ${
                          selectedTargetClass === 'auto'
                            ? 'bg-slate-800 text-cyan-400 shadow-sm border border-slate-700/60'
                            : 'text-slate-400 hover:text-slate-200'
                        }`}
                      >
                        Argmax ({result.predicted_class.slice(0, 4)}.)
                      </button>
                      <button
                        role="radio"
                        aria-checked={selectedTargetClass === 'normal'}
                        onClick={() => runInference(selectedFile, 'normal')}
                        disabled={isExplaining}
                        className={`py-1.5 px-2 rounded-lg font-medium text-center transition-colors ${
                          selectedTargetClass === 'normal'
                            ? 'bg-slate-800 text-emerald-400 shadow-sm border border-slate-700/60'
                            : 'text-slate-400 hover:text-slate-200'
                        }`}
                      >
                        Normal
                      </button>
                      <button
                        role="radio"
                        aria-checked={selectedTargetClass === 'osteopenia'}
                        onClick={() => runInference(selectedFile, 'osteopenia')}
                        disabled={isExplaining}
                        className={`py-1.5 px-2 rounded-lg font-medium text-center transition-colors ${
                          selectedTargetClass === 'osteopenia'
                            ? 'bg-slate-800 text-amber-400 shadow-sm border border-slate-700/60'
                            : 'text-slate-400 hover:text-slate-200'
                        }`}
                      >
                        Osteopenia
                      </button>
                      <button
                        role="radio"
                        aria-checked={selectedTargetClass === 'osteoporosis'}
                        onClick={() => runInference(selectedFile, 'osteoporosis')}
                        disabled={isExplaining}
                        className={`py-1.5 px-2 rounded-lg font-medium text-center transition-colors ${
                          selectedTargetClass === 'osteoporosis'
                            ? 'bg-slate-800 text-rose-400 shadow-sm border border-slate-700/60'
                            : 'text-slate-400 hover:text-slate-200'
                        }`}
                      >
                        Osteoporosis
                      </button>
                    </div>
                  </div>

                  {/* Metadata stats row */}
                  <div className="pt-3 border-t border-slate-800/80 flex items-center justify-between text-[11px] font-mono text-slate-500">
                    <span>Latency:</span>
                    <span className="tabular-nums text-slate-300">{result.inference_ms} ms</span>
                  </div>
                </div>
              ) : null}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
