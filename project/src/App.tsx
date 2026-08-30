import { ToastProvider } from "@/components/Toast";
import { NavProvider, useNav } from "@/components/NavContext";
import Sidebar from "@/components/Sidebar";
import Topbar from "@/components/Topbar";
import Overview from "@/pages/Overview";
import TransactionDetector from "@/pages/TransactionDetector";
import RedTeamSimulator from "@/pages/RedTeamSimulator";
import AttackIntelligence from "@/pages/AttackIntelligence";
import ModelPerformance from "@/pages/ModelPerformance";
import Explainability from "@/pages/Explainability";
import SystemHealth from "@/pages/SystemHealth";

function PageRouter() {
  const { page } = useNav();
  switch (page) {
    case "overview": return <Overview />;
    case "detector": return <TransactionDetector />;
    case "redteam": return <RedTeamSimulator />;
    case "intelligence": return <AttackIntelligence />;
    case "performance": return <ModelPerformance />;
    case "explainability": return <Explainability />;
    case "health": return <SystemHealth />;
    default: return <Overview />;
  }
}

function Shell() {
  const { collapsed } = useNav();
  return (
    <div className="min-h-screen bg-[#070b14] grid-bg">
      <Sidebar />
      <div
        className={`transition-[margin] duration-300 ease-out ${collapsed ? "ml-[72px]" : "ml-[248px]"}`}
      >
        <Topbar />
        <main className="px-6 py-8 max-w-[1600px] mx-auto">
          <div key={Math.random()} className="animate-fade-in">
            <PageRouter />
          </div>
        </main>
      </div>
    </div>
  );
}

export default function App() {
  return (
    <ToastProvider>
      <NavProvider>
        <Shell />
      </NavProvider>
    </ToastProvider>
  );
}
