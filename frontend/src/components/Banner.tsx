import React from 'react';

export const Banner: React.FC = () => {
  return (
    <aside
      aria-label="Clinical disclaimer"
      className="bg-[#0b1017] dark:bg-[#070b10] light:bg-slate-100 border-b border-slate-800/80 text-slate-400 text-xs px-4 py-1.5 flex items-center justify-center text-center tracking-tight"
    >
      <div className="flex items-center gap-2">
        <span className="w-1.5 h-1.5 rounded-full bg-cyan-400/80 shrink-0" aria-hidden="true" />
        <span>
          <strong className="font-semibold text-slate-200">Research prototype</strong> — not for diagnosis or clinical decision-making.
        </span>
      </div>
    </aside>
  );
};
