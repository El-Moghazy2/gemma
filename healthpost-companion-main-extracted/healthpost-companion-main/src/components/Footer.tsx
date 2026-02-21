import { Heart } from "lucide-react";

const Footer = () => (
  <footer className="w-full border-t border-border py-8 mt-12">
    <div className="max-w-5xl mx-auto px-4">
      <div className="flex flex-col md:flex-row items-center justify-between gap-3">
        <p className="text-sm text-muted-foreground">
          <span className="font-semibold text-foreground">HealthPost</span> — Supporting CHWs to deliver better care
        </p>
        <p className="text-sm text-muted-foreground flex items-center gap-1.5">
          Built with <Heart className="w-3.5 h-3.5 text-destructive inline" /> for the MedGemma Impact Challenge 2025
        </p>
      </div>
    </div>
  </footer>
);

export default Footer;