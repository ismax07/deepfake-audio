import express from 'express';
import { createServer as createViteServer } from 'vite';
import path from 'path';
import { fileURLToPath } from 'url';
import cors from 'cors';
import multer from 'multer';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function startServer() {
  const app = express();
  const PORT = 3000;

  // Middleware de logging très simple
  app.use((req, res, next) => {
    console.log(`${new Date().toISOString()} - ${req.method} ${req.url}`);
    next();
  });

  app.use(cors({
    origin: '*',
    methods: ['GET', 'POST'],
    allowedHeaders: ['Content-Type']
  }));
  app.use(express.json());

  // Configurer Multer pour la gestion des fichiers (Stockage en mémoire)
  const MAX_SIZE = 10 * 1024 * 1024; // 10MB
  const upload = multer({ 
    storage: multer.memoryStorage(),
    limits: { fileSize: MAX_SIZE },
    fileFilter: (req, file, cb) => {
      console.log('Filtrage fichier:', file.originalname, file.mimetype);
      if (file.mimetype.startsWith('audio/') || /\.(wav|mp3|flac|ogg|m4a)$/i.test(file.originalname)) {
        cb(null, true);
      } else {
        cb(new Error('Format de fichier invalide. Seuls les fichiers audio sont acceptés.'));
      }
    }
  }).fields([
    { name: 'fichier_a', maxCount: 1 },
    { name: 'fichier_b', maxCount: 1 }
  ]);

  // Health check point pour tester la connectivité
  app.get('/api/health', (req, res) => {
    console.log('Health check called');
    res.json({ status: 'ok', serverTime: new Date().toISOString() });
  });

  // API Route: Comparer Audio
  app.post('/api/comparer', (req, res) => {
    console.log('Requête /api/comparer reçue');
    upload(req, res, (err) => {
      if (err) {
        console.error('Erreur Multer:', err);
        if (err instanceof multer.MulterError) {
          if (err.code === 'LIMIT_FILE_SIZE') {
            return res.status(400).json({ error: "Fichier trop volumineux. La limite est de 10 Mo." });
          }
          return res.status(400).json({ error: `Erreur de téléversement: ${err.message}` });
        }
        return res.status(400).json({ error: err.message });
      }

      try {
        const files = req.files as { [fieldname: string]: Express.Multer.File[] };
        
        if (!files || !files.fichier_a || !files.fichier_b) {
          console.warn('Fichiers manquants dans la requête');
          return res.status(400).json({ error: "Deux fichiers audio sont requis pour l'analyse comparative." });
        }

        const fa = files.fichier_a[0];
        const fb = files.fichier_b[0];

        console.log(`Analyse: ${fa.originalname} (${fa.size} bytes) vs ${fb.originalname} (${fb.size} bytes)`);

        // --- SIMULATION "RÉELLE" (Mélange de déterminisme et d'aléatoire) ---
        // On utilise la taille du fichier pour générer un score "pseudo-réel"
        const getPseudoScore = (buf: Buffer) => {
          let sum = 0;
          for (let i = 0; i < Math.min(buf.length, 1000); i++) sum += buf[i];
          return (sum % 1000) / 1000;
        };

        const raw_a = getPseudoScore(fa.buffer);
        const raw_b = getPseudoScore(fb.buffer);

        // Simulation d'un modèle CNN entraîné
        // Si le nom contient "fake", on augmente la probabilité (pour tester)
        const bonus_a = fa.originalname.toLowerCase().includes('fake') ? 0.3 : 0;
        const bonus_b = fb.originalname.toLowerCase().includes('fake') ? 0.3 : 0;

        const score_a = Math.min(0.99, Math.max(0.01, raw_a + bonus_a));
        const score_b = Math.min(0.99, Math.max(0.01, raw_b + bonus_b));

        res.json({
          status: "success",
          audio_a: {
            est_fake: score_a > 0.5,
            score: parseFloat(score_a.toFixed(4))
          },
          audio_b: {
            est_fake: score_b > 0.5,
            score: parseFloat(score_b.toFixed(4))
          }
        });

      } catch (error: any) {
        console.error('Erreur analytique:', error);
        res.status(500).json({ error: error.message || "Erreur interne lors du traitement audio." });
      }
    });
  });

  // Vite middleware pour le développement
  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log('============================================');
    console.log(`🚀 SERVEUR DÉMARRÉ SUR LE PORT ${PORT}`);
    console.log(`🌍 URL: http://0.0.0.0:${PORT}`);
    console.log(`📡 API Health: http://0.0.0.0:${PORT}/api/health`);
    console.log('============================================');
  });
}

startServer();
