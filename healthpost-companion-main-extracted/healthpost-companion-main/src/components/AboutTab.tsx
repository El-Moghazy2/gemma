import { Mic, Eye, Brain, Pill, AlertTriangle, Shield } from "lucide-react";

const features = [
  { icon: Mic, title: "Voice Intake", model: "MedASR", desc: "Transcribe spoken symptoms into structured clinical text" },
  { icon: Eye, title: "Image Analysis", model: "MedGemma Vision", desc: "Analyze skin conditions, wounds, and other visual findings" },
  { icon: Brain, title: "Diagnosis & Treatment", model: "MedGemma Text", desc: "Generate differential diagnosis and evidence-based treatment plans" },
  { icon: Pill, title: "Drug Safety", model: "DDInter API", desc: "Check for drug-drug interactions and contraindications" },
  { icon: AlertTriangle, title: "Referral Guidance", model: "Rule-based", desc: "Assess danger signs and determine if facility referral is needed" },
];

const pipeline = [
  { step: "01", title: "Intake", desc: "Patient symptoms collected via text, voice, or pre-filled demo scenarios" },
  { step: "02", title: "Image Analysis", desc: "Medical images processed by MedGemma Vision for visual findings" },
  { step: "03", title: "Diagnosis", desc: "MedGemma Text generates differential diagnosis and treatment plan" },
  { step: "04", title: "Drug Safety", desc: "Current medications checked against DDInter database for interactions" },
  { step: "05", title: "Safety Assessment", desc: "Rule-based engine evaluates referral criteria and danger signs" },
];

const AboutTab = () => (
  <div className="space-y-8">
    {/* Hero */}
    <div className="panel shadow-elevated">
      <h2 className="text-2xl font-display font-bold text-foreground mb-3">About HealthPost</h2>
      <p className="text-muted-foreground leading-relaxed">
        HealthPost empowers <strong className="text-foreground">Community Health Workers (CHWs)</strong> in low-resource settings by
        providing AI-assisted clinical decision support powered by <strong className="text-foreground">Google's MedGemma</strong> model.
        By combining voice-based intake, medical image analysis, intelligent diagnosis, and drug safety
        checking into a single workflow, HealthPost helps CHWs deliver faster, more accurate care at the
        point of need.
      </p>
    </div>

    {/* Feature Cards */}
    <div>
      <h3 className="text-lg font-display font-semibold text-foreground mb-4">Feature Matrix</h3>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {features.map((f) => (
          <div key={f.title} className="panel group">
            <div className="flex items-start gap-3">
              <div className="w-10 h-10 rounded-xl bg-primary/10 flex items-center justify-center shrink-0 group-hover:bg-primary/15 transition-colors">
                <f.icon className="w-5 h-5 text-primary" />
              </div>
              <div className="min-w-0">
                <h4 className="font-semibold text-foreground text-sm">{f.title}</h4>
                <span className="text-xs text-primary font-medium">{f.model}</span>
                <p className="text-xs text-muted-foreground mt-1 leading-relaxed">{f.desc}</p>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>

    {/* Pipeline */}
    <div>
      <h3 className="text-lg font-display font-semibold text-foreground mb-4">Architecture Pipeline</h3>
      <div className="panel p-0 overflow-hidden">
        {pipeline.map((p, i) => (
          <div key={p.step} className={`flex items-start gap-4 p-5 ${i < pipeline.length - 1 ? "border-b border-border" : ""} hover:bg-accent/30 transition-colors`}>
            <span className="text-xs font-bold text-primary bg-primary/10 rounded-lg w-9 h-9 flex items-center justify-center shrink-0 font-display">
              {p.step}
            </span>
            <div>
              <h4 className="font-semibold text-foreground text-sm">{p.title}</h4>
              <p className="text-xs text-muted-foreground mt-0.5 leading-relaxed">{p.desc}</p>
            </div>
          </div>
        ))}
      </div>
    </div>

    {/* Technical Details */}
    <div className="panel">
      <h3 className="text-lg font-display font-semibold text-foreground mb-3">Technical Details</h3>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {[
          { label: "4-bit quantization", desc: "MedGemma optimized for consumer-grade hardware" },
          { label: "Pydantic models", desc: "Structured output parsing ensures consistent results" },
          { label: "Gradio UI", desc: "Accessible interface designed for low-bandwidth environments" },
          { label: "MedGemma", desc: "Google's medical model fine-tuned for clinical reasoning" },
        ].map((t) => (
          <div key={t.label} className="flex items-start gap-2.5 p-3 rounded-xl bg-accent/30">
            <div className="w-1.5 h-1.5 rounded-full bg-primary mt-2 shrink-0" />
            <div>
              <span className="text-sm font-medium text-foreground">{t.label}</span>
              <p className="text-xs text-muted-foreground">{t.desc}</p>
            </div>
          </div>
        ))}
      </div>
    </div>

    {/* Disclaimer */}
    <div className="panel border-clinical-warning/30 bg-clinical-warning/5">
      <div className="flex items-start gap-3">
        <Shield className="w-5 h-5 text-clinical-warning shrink-0 mt-0.5" />
        <div>
          <h4 className="font-semibold text-foreground text-sm mb-1">⚠️ Important Disclaimer</h4>
          <p className="text-sm text-muted-foreground leading-relaxed">
            This system is designed to <strong className="text-foreground">support, not replace</strong>, clinical judgment.
            All AI-generated recommendations should be reviewed by a qualified healthcare professional.
            HealthPost is a decision-support tool and does not provide definitive medical diagnoses.
            Always follow local clinical guidelines and protocols.
          </p>
        </div>
      </div>
    </div>
  </div>
);

export default AboutTab;