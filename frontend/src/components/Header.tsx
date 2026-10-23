import React from 'react';
import { Sun, Moon } from 'lucide-react';
import type { ActiveTab, HealthResponse } from '../types';

interface HeaderProps {
  activeTab: ActiveTab;
  onTabChange: (tab: ActiveTab) => void;
  health: HealthResponse | null;
  healthLoading: boolean;
  isDark: boolean;
  onToggleTheme: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  activeTab,
  onTabChange,
  health,
  healthLoading,
  isDark,
  onToggleTheme,
}) => {
  const isOnline = !!health?.model_loaded;
  const deviceTooltip = health
    ? `Backend device: ${health.device.toUpperCase()} (Model loaded)`
    : healthLoading
    ? 'Checking model service...'
    : 'Backend service offline';

  return (
    <header className="bg-[#0b0f14]/90 dark:bg-[#070a0e]/95 backdrop-blur-md border-b border-slate-800/80 sticky top-0 z-40 transition-colors">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Brand & Subtitle */}
          <div className="flex items-center gap-3">
            {/* Minimalist reticle / bone logo mark */}
            <div className="w-8 h-8 rounded-lg bg-slate-900 border border-slate-800 flex items-center justify-center shrink-0">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <rect x="3" y="3" width="18" height="18" rx="4" stroke="#00e5ff" strokeWidth="1.5" />
                <circle cx="12" cy="12" r="3" fill="#00e5ff" opacity="0.9" />
                <path d="M12 4V8M12 16V20M4 12H8M16 12H20" stroke="#00e5ff" strokeWidth="1.5" strokeLinecap="round" />
              </svg>
            </div>
            <div>
              <div className="flex items-center gap-2.5">
                <span className="font-semibold text-slate-100 text-lg tracking-tight">
                  OsteoKnee
                </span>
              </div>
              <p className="text-xs text-slate-400 hidden sm:block font-normal">
                Knee X-ray bone density classifier — research demo
              </p>
            </div>
          </div>

          {/* Navigation Segmented Control */}
          <nav
            role="tablist"
            aria-label="Navigation"
            className="flex items-center p-1 bg-slate-900/90 rounded-lg border border-slate-800"
          >
            <button
              role="tab"
              aria-selected={activeTab === 'analyze'}
              onClick={() => onTabChange('analyze')}
              className={`px-3.5 py-1.5 rounded-md text-xs sm:text-sm font-medium transition-all ${
                activeTab === 'analyze'
                  ? 'bg-slate-800 text-cyan-400 shadow-sm border border-slate-700/60'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Analyze
            </button>
            <button
              role="tab"
              aria-selected={activeTab === 'evaluation'}
              onClick={() => onTabChange('evaluation')}
              className={`px-3.5 py-1.5 rounded-md text-xs sm:text-sm font-medium transition-all ${
                activeTab === 'evaluation'
                  ? 'bg-slate-800 text-cyan-400 shadow-sm border border-slate-700/60'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Model
            </button>
          </nav>

          {/* Status Dot & Theme Toggle */}
          <div className="flex items-center gap-4">
            {/* Status dot: Model online / Model offline with device in tooltip */}
            <div
              className="flex items-center gap-2 cursor-help text-xs select-none"
              title={deviceTooltip}
            >
              <span
                className={`w-2 h-2 rounded-full ${
                  isOnline
                    ? 'bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.6)]'
                    : 'bg-rose-500 shadow-[0_0_8px_rgba(244,63,94,0.6)]'
                }`}
                aria-hidden="true"
              />
              <span className="font-medium text-slate-300">
                {isOnline ? 'Model online' : 'Model offline'}
              </span>
            </div>

            {/* Theme Toggle Button */}
            <button
              onClick={onToggleTheme}
              className="p-1.5 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors"
              title={isDark ? 'Switch to light theme' : 'Switch to dark theme'}
              aria-label="Toggle theme"
            >
              {isDark ? <Sun className="w-4 h-4 text-cyan-400" /> : <Moon className="w-4 h-4 text-slate-400" />}
            </button>
          </div>
        </div>
      </div>
    </header>
  );
};
