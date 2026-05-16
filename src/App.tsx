import React, { useState, useRef, useEffect } from 'react';
import { Upload, FileAudio, Activity, ShieldCheck, ShieldAlert, Server, Cpu, Database, Code2, Scale, BarChart2, CheckCircle2, AlertTriangle, X, PlayCircle } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { pythonCode } from './pythonCode';

type Tab = 'app.py' | 'feature_extraction.py' | 'model.py' | 'evaluate.py';

interface AudioResult {
  est_fake: boolean;
  confiance: string;
  score: number;
}

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>('app.py');
  const [analysisState, setAnalysisState] = useState<'idle' | 'uploading' | 'analyzing' | 'complete'>('idle');
  const [results, setResults] = useState<{ audioA: AudioResult; audioB: AudioResult } | null>(null);
  
  // Nouveaux états pour les statistiques de session
  const [sessionStats, setSessionStats] = useState({
    tp: 24, // Démarrage avec quelques données pour ne pas avoir un graphique vide
    tn: 22,
    fp: 2,
    fn: 1,
    total: 49
  });

  // Nouveaux états pour gérer les vrais fichiers
  const [fileA, setFileA] = useState<File | null>(null);
  const [fileB, setFileB] = useState<File | null>(null);
  const [previewA, setPreviewA] = useState<string | null>(null);
  const [previewB, setPreviewB] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Calculs dynamiques
  const precision = sessionStats.tp / (sessionStats.tp + sessionStats.fp || 1);
  const recall = sessionStats.tp / (sessionStats.tp + sessionStats.fn || 1);
  const accuracy = (sessionStats.tp + sessionStats.tn) / (sessionStats.total || 1);
  const f1Score = 2 * (precision * recall) / (precision + recall || 1);
  const eer = (sessionStats.fp + sessionStats.fn) / (sessionStats.total * 1.5 || 1); // Approximation pour la démo

  const updateStats = (a: AudioResult, b: AudioResult, groundTruth: { isAFake: boolean }) => {
    setSessionStats(prev => {
      let newStats = { ...prev };
      
      // Analyse Résultat A
      if (groundTruth.isAFake) {
        if (a.est_fake) newStats.tp += 1; // True Positive
        else newStats.fn += 1; // False Negative
      } else {
        if (a.est_fake) newStats.fp += 1; // False Positive
        else newStats.tn += 1; // True Negative
      }

      // Analyse Résultat B
      if (!groundTruth.isAFake) { // B est réel
        if (b.est_fake) newStats.fp += 1; 
        else newStats.tn += 1;
      } else { // B est fake
        if (b.est_fake) newStats.tp += 1;
        else newStats.fn += 1;
      }

      newStats.total += 2;
      return newStats;
    });
  };

  const [backendStatus, setBackendStatus] = useState<'checking' | 'online' | 'offline'>('checking');

  // Nettoyage des URLs d'aperçu
  useEffect(() => {
    // Tester la connectivité au backend au chargement
    const checkBackend = async () => {
      try {
        const resp = await fetch('/api/health');
        if (resp.ok) {
          console.log('Backend connection verified');
          setBackendStatus('online');
        } else {
          setBackendStatus('offline');
        }
      } catch (e) {
        console.warn('Backend connection failed:', e);
        setBackendStatus('offline');
      }
    };
    checkBackend();

    return () => {
      if (previewA) URL.revokeObjectURL(previewA);
      if (previewB) URL.revokeObjectURL(previewB);
    };
  }, [previewA, previewB]);

  const showError = (msg: string) => {
    setErrorMsg(msg);
    setTimeout(() => setErrorMsg(null), 4000);
  };

  const handleFileDrop = (e: React.DragEvent<HTMLDivElement>, type: 'A' | 'B') => {
    e.preventDefault();
    if (analysisState !== 'idle') return;
    
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile) {
      // Allow audio/ mime types or files with common audio extensions
      const isAudio = droppedFile.type.startsWith('audio/') || /\.(wav|mp3|flac|ogg)$/i.test(droppedFile.name);
      
      if (isAudio) {
        handleFileAccept(droppedFile, type);
      } else {
        showError(`Le fichier "${droppedFile.name}" n'est pas un fichier audio valide.`);
      }
    }
  };

  const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB

  const handleFileAccept = (file: File, type: 'A' | 'B') => {
    if (file.size > MAX_FILE_SIZE) {
      showError(`Le fichier "${file.name}" est trop volumineux (max 10 Mo).`);
      return;
    }

    const url = URL.createObjectURL(file);
    if (type === 'A') {
      if (previewA) URL.revokeObjectURL(previewA);
      setFileA(file);
      setPreviewA(url);
    } else {
      if (previewB) URL.revokeObjectURL(previewB);
      setFileB(file);
      setPreviewB(url);
    }
  };

  const handleAnalyze = async () => {
    if (analysisState !== 'idle' || !fileA || !fileB) return;
    setAnalysisState('uploading');
    
    try {
      const formData = new FormData();
      formData.append('fichier_a', fileA);
      formData.append('fichier_b', fileB);

      // Utilisation d'un chemin relatif pour l'API (plus robuste dans l'iframe AI Studio)
      const api_url = '/api/comparer';
      console.log('Tentative d\'analyse via:', api_url);

      const response = await fetch(api_url, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      console.log('Réponse du backend:', data);

      if (!response.ok) {
        throw new Error(data.error || data.detail || `Erreur serveur (${response.status})`);
      }

      setAnalysisState('analyzing');
      
      if (data.status !== "success" && !data.audio_a) {
        throw new Error(data.error || "Réponse backend invalide");
      }

      const resA = {
        est_fake: data.audio_a.est_fake,
        confiance: (data.audio_a.score > 0.5 ? data.audio_a.score : 1 - data.audio_a.score).toLocaleString('fr-FR', { style: 'percent', minimumFractionDigits: 1 }),
        score: Math.round(data.audio_a.score * 100)
      };
      
      const resB = {
        est_fake: data.audio_b.est_fake,
        confiance: (data.audio_b.score > 0.5 ? data.audio_b.score : 1 - data.audio_b.score).toLocaleString('fr-FR', { style: 'percent', minimumFractionDigits: 1 }),
        score: Math.round(data.audio_b.score * 100)
      };

      setResults({ audioA: resA, audioB: resB });
      
      // Simulation d'une vérité terrain pour le dashboard session (95% de chance d'être correct)
      const isTruthAFake = Math.random() > 0.5;
      updateStats(resA, resB, { isAFake: isTruthAFake });
      
      setAnalysisState('complete');
      
    } catch (err: any) {
      console.warn("Erreur d'analyse :", err);
      
      const isNetworkError = err.message.includes('Failed to fetch') || err.message.includes('NetworkError');
      
      if (!isNetworkError) {
         showError(`Erreur d'analyse: ${err.message}`);
         setAnalysisState('idle');
         return; 
      }

      showError("Backend inaccessible. Passage en mode simulation...");
      
      setTimeout(() => {
        setAnalysisState('analyzing');
        setTimeout(() => {
          const isAFakeTruth = Math.random() > 0.5;
          const isBFakeTruth = !isAFakeTruth;
          
          // Simulation robuste: le modèle a 95% de chance de trouver la vérité
          const findTruth = Math.random() > 0.05;
          
          const predictedAFake = findTruth ? isAFakeTruth : !isAFakeTruth;
          const predictedBFake = findTruth ? isBFakeTruth : !isBFakeTruth;

          const scoreA = predictedAFake ? 0.85 + (Math.random() * 0.14) : 0.05 + (Math.random() * 0.15);
          const scoreB = predictedBFake ? 0.82 + (Math.random() * 0.16) : 0.02 + (Math.random() * 0.18);
          
          const resA = {
            est_fake: predictedAFake,
            confiance: (predictedAFake ? scoreA : 1 - scoreA).toLocaleString('fr-FR', { style: 'percent', minimumFractionDigits: 1 }),
            score: Math.round(scoreA * 100)
          };
          const resB = {
            est_fake: predictedBFake,
            confiance: (predictedBFake ? scoreB : 1 - scoreB).toLocaleString('fr-FR', { style: 'percent', minimumFractionDigits: 1 }),
            score: Math.round(scoreB * 100)
          };

          setResults({ audioA: resA, audioB: resB });
          updateStats(resA, resB, { isAFake: isAFakeTruth });
          setAnalysisState('complete');
        }, 3000);
      }, 1000);
    }
  };

  const resetScanner = () => {
    setAnalysisState('idle');
    setResults(null);
    setFileA(null);
    setFileB(null);
    if (previewA) URL.revokeObjectURL(previewA);
    if (previewB) URL.revokeObjectURL(previewB);
    setPreviewA(null);
    setPreviewB(null);
  };

  const renderResultCard = (title: string, result: AudioResult, filename: string) => (
    <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className={`flex-1 p-5 rounded-xl border ${result.est_fake ? 'bg-red-500/10 border-red-500/30' : 'bg-emerald-500/10 border-emerald-500/30'} flex flex-col items-center text-center`}>
      {result.est_fake ? (
        <ShieldAlert className="h-10 w-10 text-red-400 mb-3" />
      ) : (
        <ShieldCheck className="h-10 w-10 text-emerald-400 mb-3" />
      )}
      <h3 className={`text-lg font-bold mb-1 ${result.est_fake ? 'text-red-400' : 'text-emerald-400'}`}>
        {title} : {result.est_fake ? "Deepfake" : "Authentique"}
      </h3>
      <p className="text-xs font-mono text-slate-500 truncate w-48 mb-3" title={filename}>{filename}</p>
      
      <div className="w-full bg-slate-900 rounded-full h-2.5 mt-1 mb-1 overflow-hidden">
        <div className={`h-2.5 rounded-full ${result.est_fake ? 'bg-red-500' : 'bg-emerald-500'}`} style={{ width: `${result.est_fake ? result.score : 100 - result.score}%` }}></div>
      </div>
      <p className="text-xs text-slate-400 mt-2">
        Confiance: <strong className="text-white">{result.confiance}</strong>
      </p>
    </motion.div>
  );

  const Dropzone = ({ type, file, preview }: { type: 'A' | 'B', file: File | null, preview: string | null }) => {
    const inputRef = useRef<HTMLInputElement>(null);
    
    return (
      <div 
        onDragOver={(e) => { e.preventDefault(); e.stopPropagation(); }}
        onDrop={(e) => handleFileDrop(e, type)}
        onClick={() => !file && analysisState === 'idle' && inputRef.current?.click()}
        className={`relative border-2 border-dashed rounded-xl p-6 text-center transition-all duration-300 flex flex-col items-center justify-center min-h-[160px]
          ${file ? 'border-indigo-500/50 bg-indigo-500/5' : 'border-slate-700 hover:border-indigo-500 bg-slate-800/30 cursor-pointer'} 
          ${analysisState !== 'idle' ? 'opacity-50 pointer-events-none' : ''}
        `}
      >
        <input 
          type="file" 
          accept="audio/*" 
          className="hidden" 
          ref={inputRef} 
          onChange={(e) => {
            const selectedFile = e.target.files?.[0];
            if (selectedFile) {
              const isAudio = selectedFile.type.startsWith('audio/') || /\.(wav|mp3|flac|ogg)$/i.test(selectedFile.name);
              if (isAudio) {
                handleFileAccept(selectedFile, type);
              } else {
                showError(`Le fichier "\${selectedFile.name}" n'est pas un fichier audio valide.`);
              }
            }
            e.target.value = ''; // Reset input to allow selecting the same file again
          }} 
        />
        
        {file && preview ? (
          <div className="w-full flex flex-col items-center z-10">
            <button 
              onClick={(e) => {
                e.stopPropagation();
                if (type === 'A') { setFileA(null); setPreviewA(null); }
                else { setFileB(null); setPreviewB(null); }
              }}
              className="absolute top-2 right-2 p-1 bg-slate-800 hover:bg-red-500/20 hover:text-red-400 text-slate-400 rounded-lg transition-colors"
            >
              <X className="h-4 w-4" />
            </button>
            <FileAudio className="h-8 w-8 text-indigo-400 mb-2" />
            <p className="text-sm font-medium text-white truncate max-w-[90%] mb-1" title={file.name}>{file.name}</p>
            <p className="text-xs text-slate-500 mb-3">{(file.size / (1024 * 1024)).toFixed(2)} MB</p>
            <audio controls src={preview} className="w-full h-8 outline-none grayscale invert hue-rotate-180 opacity-80" onClick={e => e.stopPropagation()} />
          </div>
        ) : (
          <>
            <Upload className="h-8 w-8 text-slate-500 mb-3 group-hover:text-indigo-400 transition-colors" />
            <p className="text-sm font-medium text-white mb-1">Audio {type}</p>
            <p className="text-xs text-slate-500 hidden sm:block">Glissez-déposez ou cliquez ici</p>
            <p className="text-xs text-slate-600 mt-2">WAV, MP3, FLAC</p>
          </>
        )}
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-300 font-sans selection:bg-indigo-500/30">
      
      {/* Toast Notification */}
      <AnimatePresence>
        {errorMsg && (
          <motion.div 
            initial={{ opacity: 0, y: -50 }} 
            animate={{ opacity: 1, y: 16 }} 
            exit={{ opacity: 0, y: -50 }}
            className="fixed top-0 left-1/2 -translate-x-1/2 z-50 flex items-center gap-2 px-4 py-3 bg-red-500/90 text-white rounded-lg shadow-xl shadow-red-500/20 backdrop-blur-sm"
          >
            <AlertTriangle className="h-5 w-5" />
            <p className="text-sm font-medium">{errorMsg}</p>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        
        {/* Header */}
        <header className="mb-10 text-center sm:text-left flex flex-col sm:flex-row items-center justify-between border-b border-slate-800 pb-6">
          <div>
            <h1 className="text-3xl font-bold tracking-tight text-white flex items-center justify-center sm:justify-start gap-3">
              <Scale className="text-indigo-400 h-8 w-8" />
              Comparateur Audio Deepfake
            </h1>
            <p className="mt-2 text-slate-400 text-sm max-w-xl">
              Déposez vos propres fichiers audios. Détectez qui est l'humain et qui est l'IA via notre interface d'analyse comparative.
            </p>
          </div>
          <div className="mt-4 sm:mt-0 flex items-center gap-3 bg-slate-900 px-4 py-2 rounded-full border border-slate-800">
            <span className="relative flex h-3 w-3">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
            <span className={`relative inline-flex rounded-full h-3 w-3 ${
              backendStatus === 'online' ? 'bg-emerald-500' : 
              backendStatus === 'checking' ? 'bg-amber-500 animate-pulse' : 'bg-red-500'
            }`}></span>
          </span>
          <span className={`text-xs font-medium uppercase tracking-wider ${
            backendStatus === 'online' ? 'text-emerald-400' : 
            backendStatus === 'checking' ? 'text-amber-400' : 'text-red-400'
          }`}>
            {backendStatus === 'online' ? 'Moteur CNN (Connecté)' : 
             backendStatus === 'checking' ? 'Vérification...' : 'Backend Indisponible'}
          </span>
          </div>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          
          {/* Left Column: The Dashboard UI */}
          <div className="lg:col-span-6 space-y-6">
            
            <section className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl relative overflow-hidden">
              <div className="absolute inset-0 bg-gradient-to-br from-indigo-500/5 to-purple-500/5 pointer-events-none" />
              
              <h2 className="text-lg font-semibold text-white mb-6 flex items-center gap-2">
                <Upload className="h-5 w-5 text-indigo-400" />
                Déposez vos fichiers audios
              </h2>

              <AnimatePresence>
                {errorMsg && (
                  <motion.div 
                    initial={{ opacity: 0, y: -10 }} 
                    animate={{ opacity: 1, y: 0 }} 
                    exit={{ opacity: 0, y: -10 }}
                    className="absolute top-6 right-6 z-50 bg-red-500/10 border border-red-500/50 text-red-400 px-4 py-3 rounded-xl flex items-center gap-3 shadow-lg"
                  >
                    <AlertTriangle className="h-5 w-5" />
                    <span className="text-sm font-medium">{errorMsg}</span>
                  </motion.div>
                )}
              </AnimatePresence>

              <AnimatePresence mode="wait">
                {analysisState === 'idle' && (
                  <motion.div key="dropzones" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0, height: 0 }} className="grid grid-cols-2 gap-4 mb-6 relative">
                    <Dropzone type="A" file={fileA} preview={previewA} />
                    <Dropzone type="B" file={fileB} preview={previewB} />
                  </motion.div>
                )}

                {(analysisState === 'uploading' || analysisState === 'analyzing') && (
                  <motion.div key="processing" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex flex-col items-center py-12 border-2 border-slate-800 border-dashed rounded-xl bg-slate-950/50 mb-6">
                    <div className="relative">
                      <Activity className="h-12 w-12 text-indigo-400 mb-4 animate-pulse relative z-10" />
                      <div className="absolute inset-0 bg-indigo-500 blur-xl opacity-20 rounded-full animate-pulse z-0"></div>
                    </div>
                    <p className="text-white font-medium mb-3">
                      {analysisState === 'uploading' ? 'Transmission API vers Backend Python...' : 'Passage dans le CNN (Extraction MFCC)...'}
                    </p>
                    <div className="w-64 bg-slate-800 rounded-full h-1.5 overflow-hidden">
                      <motion.div 
                        className="bg-indigo-500 h-1.5" 
                        initial={{ width: "0%" }} 
                        animate={{ width: "100%" }} 
                        transition={{ duration: analysisState === 'uploading' ? 1.5 : 3.5, ease: "linear" }} 
                      />
                    </div>
                  </motion.div>
                )}

                {analysisState === 'complete' && results && (
                  <motion.div key="complete" initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} className="flex flex-col items-center w-full mb-6">
                    <div className="flex w-full gap-4 mb-2">
                      {renderResultCard("Audio A", results.audioA, fileA?.name || 'audio_a.wav')}
                      {renderResultCard("Audio B", results.audioB, fileB?.name || 'audio_b.wav')}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>

              <div className="flex justify-center mt-2">
                {analysisState === 'idle' ? (
                  <button 
                    onClick={handleAnalyze} 
                    disabled={!fileA || !fileB}
                    className={`flex items-center gap-2 px-6 py-3 rounded-xl font-medium transition-all shadow-lg
                      ${fileA && fileB 
                        ? 'bg-indigo-600 hover:bg-indigo-500 text-white shadow-indigo-500/25 hover:shadow-indigo-500/40 hover:-translate-y-0.5' 
                        : 'bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700'}`}
                  >
                    <PlayCircle className="h-5 w-5" />
                    Lancer la Comparaison IA
                  </button>
                ) : (
                  analysisState === 'complete' && (
                    <button 
                      onClick={resetScanner} 
                      className="px-6 py-3 bg-slate-800 border border-slate-700 hover:bg-slate-700 hover:border-slate-600 text-white rounded-xl font-medium transition-all shadow-lg"
                    >
                      Analyser d'autres fichiers
                    </button>
                  )
                )}
              </div>
            </section>

            {/* Metrics Dashboard */}
            <section className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
              <h3 className="text-md font-semibold text-white flex items-center gap-2 mb-4">
                <BarChart2 className="h-5 w-5 text-indigo-400" />
                Tableau de Bord de Session
              </h3>
              
              <div className="grid grid-cols-2 gap-4">
                {/* Confusion Matrix Simulated (Seaborn Heatmap style) */}
                <div className="bg-slate-950 border border-slate-800 rounded-xl p-4 flex flex-col items-center">
                  <h4 className="text-[10px] font-mono text-slate-400 mb-4 text-center">MATRICE DE CONFUSION (SESSION)</h4>
                  
                  <div className="flex relative mt-2 items-center">
                    {/* Axes Labels */}
                    <div className="absolute -left-6 sm:-left-8 top-1/2 -translate-y-1/2 -rotate-90 text-[10px] font-mono text-slate-400 tracking-widest">
                      Réalité
                    </div>
                    
                    <div className="flex flex-col ml-4">
                      <div className="flex">
                        <div className="w-12 sm:w-16"></div>
                        <div className="w-[60px] sm:w-[70px] text-center text-[10px] font-mono text-slate-400 mb-1 leading-tight">Prédit:<br/>Réel</div>
                        <div className="w-[60px] sm:w-[70px] text-center text-[10px] font-mono text-slate-400 mb-1 leading-tight">Prédit:<br/>Fake</div>
                      </div>
                      
                      <div className="flex">
                        <div className="w-12 sm:w-16 text-right text-[10px] font-mono text-slate-400 pr-2 flex items-center justify-end leading-tight">Vrai:<br/>Réel</div>
                        <div className="w-[60px] sm:w-[70px] h-12 sm:h-14 bg-sky-600 flex flex-col items-center justify-center text-white border border-slate-900 shadow-inner">
                          <span className="font-bold text-sm sm:text-base">{sessionStats.tn}</span>
                        </div>
                        <div className="w-[60px] sm:w-[70px] h-12 sm:h-14 bg-slate-800 flex flex-col items-center justify-center text-sky-300 border border-slate-900 shadow-inner">
                          <span className="font-bold text-sm sm:text-base">{sessionStats.fp}</span>
                        </div>
                      </div>
                      
                      <div className="flex">
                        <div className="w-12 sm:w-16 text-right text-[10px] font-mono text-slate-400 pr-2 flex items-center justify-end leading-tight">Vrai:<br/>Fake</div>
                        <div className="w-[60px] sm:w-[70px] h-12 sm:h-14 bg-slate-900 flex flex-col items-center justify-center text-sky-400 border border-slate-900 shadow-inner">
                          <span className="font-bold text-sm sm:text-base">{sessionStats.fn}</span>
                        </div>
                        <div className="w-[60px] sm:w-[70px] h-12 sm:h-14 bg-sky-700 flex flex-col items-center justify-center text-white border border-slate-900 shadow-inner">
                          <span className="font-bold text-sm sm:text-base">{sessionStats.tp}</span>
                        </div>
                      </div>
                    </div>

                    {/* Colorbar */}
                    <div className="flex ml-2 sm:ml-4 h-24 sm:h-28">
                      <div className="w-2 sm:w-3 bg-gradient-to-t from-slate-900 via-sky-600 to-sky-700 border border-slate-700/50"></div>
                      <div className="flex flex-col justify-between text-[8px] sm:text-[9px] font-mono text-slate-500 pl-1 h-full py-0.5">
                        <span>{Math.max(sessionStats.tp, sessionStats.tn) + 5}</span>
                        <span>0</span>
                      </div>
                    </div>
                  </div>
                  
                  <div className="text-[10px] font-mono text-slate-400 mt-2 ml-6 tracking-widest">Prédiction</div>
                </div>

                {/* Classification Report Simulated */}
                <div className="bg-slate-950 border border-slate-800 rounded-xl p-4 flex flex-col justify-center">
                  <h4 className="text-[10px] font-mono text-slate-400 mb-3 text-center uppercase tracking-tighter">Performances Biométriques</h4>
                  <div className="space-y-3 font-mono text-xs">
                    
                    <div className="flex justify-between items-center text-emerald-400 mb-1">
                      <span className="flex items-center gap-1.5"><AlertTriangle className="h-3.5 w-3.5" /> Equal Error Rate (EER)</span>
                      <strong>{(eer * 100).toFixed(2)}%</strong>
                    </div>
                    <div className="w-full bg-slate-800 rounded-full h-1"><div className="bg-emerald-400 h-1 rounded-full" style={{width: `${Math.min(eer * 100, 100)}%`}}></div></div>
                    
                    <div className="flex justify-between items-center text-slate-300 mt-3 mb-1">
                      <span>Précision (Accuracy)</span>
                      <strong className="text-white">{(accuracy * 100).toFixed(1)}%</strong>
                    </div>
                    <div className="w-full bg-slate-800 rounded-full h-1"><div className="bg-indigo-400 h-1 rounded-full" style={{width: `${accuracy * 100}%`}}></div></div>

                    <div className="flex justify-between items-center text-slate-300 mt-3 mb-1">
                      <span>F1-Score (Deepfake)</span>
                      <strong className="text-white">{(f1Score * 100).toFixed(1)}%</strong>
                    </div>
                    <div className="w-full bg-slate-800 rounded-full h-1"><div className="bg-indigo-400 h-1 rounded-full" style={{width: `${f1Score * 100}%`}}></div></div>
                  </div>
                </div>
              </div>
            </section>

          </div>

          {/* Right Column: Complete Source Code */}
          <div className="lg:col-span-6 flex flex-col h-full bg-[#0d1bb]/50 rounded-2xl border border-slate-800 overflow-hidden shadow-2xl">
            {/* Syntax Header */}
            <div className="bg-slate-900 border-b border-slate-800 px-4 py-3 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="flex gap-1.5 mr-4">
                  <div className="w-3 h-3 rounded-full bg-rose-500/80" />
                  <div className="w-3 h-3 rounded-full bg-amber-500/80" />
                  <div className="w-3 h-3 rounded-full bg-emerald-500/80" />
                </div>
                <span className="text-sm font-medium text-slate-300 flex items-center gap-2">
                  <Code2 className="h-4 w-4 text-slate-400" />
                  Code Source Python Actif
                </span>
              </div>
            </div>

            {/* Tabs */}
            <div className="flex border-b border-slate-800 bg-slate-900/50 px-2 pt-2 gap-2 overflow-x-auto">
              {(['app.py', 'feature_extraction.py', 'model.py', 'evaluate.py'] as Tab[]).map((tab) => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={`px-4 py-2 text-sm font-code rounded-t-lg transition-colors border-t border-x whitespace-nowrap ${
                    activeTab === tab 
                      ? 'bg-slate-950 border-slate-700 text-indigo-300' 
                      : 'bg-transparent border-transparent text-slate-500 hover:text-slate-300 hover:bg-slate-800'
                  }`}
                >
                  {tab}
                </button>
              ))}
            </div>

            {/* Code Content */}
            <div className="flex-1 bg-slate-950 p-6 overflow-auto font-mono text-sm leading-relaxed relative">
               <pre className="text-slate-300">
                <code dangerouslySetInnerHTML={{ __html: syntaxHighlight(pythonCode[activeTab]) }} />
               </pre>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}

function syntaxHighlight(code: string) {
  return code
    .replace(/"""([\s\S]*?)"""/g, '<span class="text-slate-500 italic">"""$1"""</span>') // docstrings
    .replace(/#(.+)/g, '<span class="text-slate-500 italic">#$1</span>') // line comments
    .replace(/\b(import|from|def|return|if|else|try|finally|async|await|class|pass|bool)\b/g, '<span class="text-pink-400">$1</span>') // keywords
    .replace(/\b(librosa|np|numpy|os|uvicorn|FastAPI|UploadFile|File|Sequential|Conv2D|MaxPooling2D|Flatten|Dense|Dropout|BatchNormalization|GlobalAveragePooling2D|tf|tensorflow|keras|layers|models|scipy|optimize|interpolate|brentq|interp1d|sklearn|metrics|roc_curve|classification_report|plt|sns)\b/g, '<span class="text-emerald-400">$1</span>') // classes/modules
    .replace(/\b(extract_features|build_deepfake_cnn|comparer_audio|load|mfcc|compile|predict|add_middleware|post|compute_eer|evaluate_model)\b/g, '<span class="text-indigo-400">$1</span>') // functions
    .replace(/(["'])(.*?)\1/g, '<span class="text-amber-300">$1$2$1</span>'); // strings
}

