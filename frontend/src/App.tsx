import React, { useState, useEffect } from 'react';
import { Banner } from './components/Banner';
import { Header } from './components/Header';
import { AnalyzeView } from './components/AnalyzeView';
import { EvaluationView } from './components/EvaluationView';
import { ErrorBoundary } from './components/ErrorBoundary';
import { API_BASE_URL } from './config';
import type { ActiveTab, HealthResponse } from './types';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<ActiveTab>('analyze');
  const [isDark, setIsDark] = useState<boolean>(() => {
    const saved = localStorage.getItem('theme');
    if (saved) return saved === 'dark';
    return true; // Dark-first theme
  });

  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthLoading, setHealthLoading] = useState<boolean>(true);

  // Sync theme with document class
  useEffect(() => {
    if (isDark) {
      document.documentElement.classList.add('dark');
      localStorage.setItem('theme', 'dark');
    } else {
      document.documentElement.classList.remove('dark');
      localStorage.setItem('theme', 'light');
    }
  }, [isDark]);

  const toggleTheme = () => {
    setIsDark((prev) => !prev);
  };

  // Health check polling
  useEffect(() => {
    let isMounted = true;
    const checkHealth = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/health`);
        if (res.ok) {
          const data: HealthResponse = await res.json();
          if (isMounted) setHealth(data);
        } else {
          if (isMounted) setHealth(null);
        }
      } catch (err) {
        if (isMounted) setHealth(null);
      } finally {
        if (isMounted) setHealthLoading(false);
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 15000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  return (
    <ErrorBoundary>
      <div className="min-h-screen flex flex-col bg-[#070a0e] dark:bg-[#070a0e] light:bg-[#f4f6f9] text-slate-100 transition-colors duration-200">
        {/* Slim Persistent Clinical Disclaimer */}
        <Banner />

        {/* Console Navigation Header */}
        <Header
          activeTab={activeTab}
          onTabChange={setActiveTab}
          health={health}
          healthLoading={healthLoading}
          isDark={isDark}
          onToggleTheme={toggleTheme}
        />

        {/* Main Lightbox / Model Workspace */}
        <main className="flex-1">
          {activeTab === 'analyze' ? (
            <AnalyzeView
              backendReady={!!health?.model_loaded}
              device={health?.device?.toUpperCase()}
            />
          ) : (
            <EvaluationView />
          )}
        </main>

        {/* Standard Single-Line Research Footer */}
        <footer className="border-t border-slate-800/80 bg-[#070a0e] py-5 text-center text-xs text-slate-500">
          <div className="max-w-7xl mx-auto px-4">
            <p>OsteoKnee · Research project by Arindam · 2026 · Not a medical device</p>
          </div>
        </footer>
      </div>
    </ErrorBoundary>
  );
};

export default App;
