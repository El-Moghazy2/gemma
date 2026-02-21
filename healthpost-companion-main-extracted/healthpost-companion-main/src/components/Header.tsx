import { Activity, Shield, Sparkles } from "lucide-react";
import { Badge } from "@/components/ui/badge";

const Header = () => (
  <header className="w-full gradient-primary py-10 relative overflow-hidden">
    {/* Decorative elements */}
    <div className="absolute inset-0 opacity-10">
      <div className="absolute top-4 left-[10%] w-32 h-32 rounded-full bg-white/20 blur-3xl" />
      <div className="absolute bottom-2 right-[15%] w-48 h-48 rounded-full bg-white/15 blur-3xl" />
    </div>

    <div className="max-w-5xl mx-auto px-4 text-center relative z-10">
      <div className="flex items-center justify-center gap-3 mb-3">
        <div className="w-12 h-12 rounded-xl bg-white/20 backdrop-blur-sm flex items-center justify-center border border-white/20 shadow-lg">
          <Activity className="w-7 h-7 text-white" />
        </div>
        <h1 className="text-3xl md:text-4xl font-display font-extrabold text-white tracking-tight">
          HealthPost
        </h1>
      </div>
      <p className="text-base md:text-lg text-white/80 font-light max-w-xl mx-auto mb-4">
        AI-powered clinical decision support — diagnosis, treatment &amp; drug safety
      </p>
      <div className="flex items-center justify-center gap-2 flex-wrap">
        <Badge variant="secondary" className="bg-white/15 text-white border-white/20 backdrop-blur-sm gap-1.5 py-1 px-3">
          <Sparkles className="w-3.5 h-3.5" />
          Powered by MedGemma
        </Badge>
        <Badge variant="secondary" className="bg-white/15 text-white border-white/20 backdrop-blur-sm gap-1.5 py-1 px-3">
          <Shield className="w-3.5 h-3.5" />
          MedGemma Impact Challenge 2025
        </Badge>
      </div>
    </div>
  </header>
);

export default Header;