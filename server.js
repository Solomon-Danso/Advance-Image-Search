const express = require('express');
const multer = require('multer');
const cors = require('cors');
const tf = require('@tensorflow/tfjs-node');
const mysql = require('mysql2/promise');
const sharp = require('sharp');
const { createCanvas, loadImage } = require('canvas');
const path = require('path');
const fs = require('fs');


const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static('public'));

// Configure multer for file uploads
const storage = multer.memoryStorage();
const upload = multer({ 
  storage: storage,
  limits: {
    fileSize: 5 * 1024 * 1024, // 5MB limit
  },
  fileFilter: (req, file, cb) => {
    if (file.mimetype.startsWith('image/')) {
      cb(null, true);
    } else {
      cb(new Error('Only image files are allowed!'), false);
    }
  }
});

// MySQL database configuration
const dbConfig = {
  host: process.env.DB_HOST || 'localhost',
  user: process.env.DB_USER || 'root',
  password: process.env.DB_PASSWORD || 'HydotTech',
  database: process.env.DB_NAME || 'ImageSearch',
  waitForConnections: true,
  connectionLimit: 10,
  queueLimit: 0
};

// Create MySQL connection pool
const pool = mysql.createPool(dbConfig);

// Initialize database tables
async function initializeDatabase() {
  try {
    const connection = await pool.getConnection();
    
    // Create designs table if it doesn't exist
    await connection.execute(`
      CREATE TABLE IF NOT EXISTS designs (
        id VARCHAR(255) PRIMARY KEY,
        image_url VARCHAR(512),
        metadata JSON,
        embedding JSON,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      )
    `);
    
    connection.release();
    console.log('Database initialized successfully');
  } catch (error) {
    console.error('Error initializing database:', error);
    throw error;
  }
}

// Advanced feature extraction without MobileNet
async function extractAdvancedFeatures(imageBuffer) {
  try {
    // Use Sharp for advanced preprocessing
    const processedBuffer = await sharp(imageBuffer)
      .resize(256, 256, {
        fit: 'cover',
        position: 'center'
      })
      .normalize()
      .sharpen()
      .removeAlpha()
      .jpeg()
      .toBuffer();

    // Extract multiple feature types
    const [colorFeatures, textureFeatures, shapeFeatures, histogramFeatures] = await Promise.all([
      extractColorMoments(processedBuffer),
      extractLBPFeatures(processedBuffer),
      extractHOGFeatures(processedBuffer),
      extractColorHistogram(processedBuffer)
    ]);

    // Combine all features
    const combinedFeatures = [
      ...colorFeatures,
      ...textureFeatures, 
      ...shapeFeatures,
      ...histogramFeatures
    ];

    console.log('Feature dimensions - Color:', colorFeatures.length, 
                'Texture:', textureFeatures.length, 
                'Shape:', shapeFeatures.length,
                'Histogram:', histogramFeatures.length,
                'Total:', combinedFeatures.length);

    return combinedFeatures;
  } catch (error) {
    console.error('Error extracting advanced features:', error);
    throw error;
  }
}

// Color Moments (mean, standard deviation, skewness for each channel)
async function extractColorMoments(imageBuffer) {
  try {
    const image = await sharp(imageBuffer).raw().toBuffer({ resolveWithObject: true });
    const { data, info } = image;
    const { width, height, channels } = info;

    const moments = [];
    
    for (let c = 0; c < channels; c++) {
      let sum = 0;
      let sumSq = 0;
      let sumCubed = 0;
      let count = 0;

      for (let i = c; i < data.length; i += channels) {
        const pixel = data[i] / 255;
        sum += pixel;
        sumSq += pixel * pixel;
        sumCubed += pixel * pixel * pixel;
        count++;
      }

      const mean = sum / count;
      const variance = (sumSq / count) - (mean * mean);
      const stdDev = Math.sqrt(Math.max(0, variance));
      const skewness = (sumCubed / count) - (3 * mean * variance) - (mean * mean * mean);

      moments.push(mean, stdDev, skewness || 0);
    }

    return moments;
  } catch (error) {
    console.error('Error extracting color moments:', error);
    return new Array(9).fill(0);
  }
}

// Local Binary Pattern (LBP) for texture features
async function extractLBPFeatures(imageBuffer) {
  try {
    const { data, info } = await sharp(imageBuffer)
      .greyscale()
      .raw()
      .toBuffer({ resolveWithObject: true });

    const { width, height } = info;
    const lbpHistogram = new Array(256).fill(0);

    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        const center = data[y * width + x];
        let pattern = 0;

        // 3x3 neighborhood
        const neighbors = [
          data[(y-1) * width + (x-1)], data[(y-1) * width + x], data[(y-1) * width + (x+1)],
          data[y * width + (x-1)], data[y * width + (x+1)],
          data[(y+1) * width + (x-1)], data[(y+1) * width + x], data[(y+1) * width + (x+1)]
        ];

        neighbors.forEach((neighbor, index) => {
          if (neighbor >= center) {
            pattern |= (1 << index);
          }
        });

        lbpHistogram[pattern]++;
      }
    }

    // Normalize histogram
    const total = lbpHistogram.reduce((sum, val) => sum + val, 0);
    return lbpHistogram.map(val => val / total);
  } catch (error) {
    console.error('Error extracting LBP features:', error);
    return new Array(256).fill(0);
  }
}

// Histogram of Oriented Gradients (HOG) for shape features
async function extractHOGFeatures(imageBuffer) {
  try {
    const { data, info } = await sharp(imageBuffer)
      .greyscale()
      .raw()
      .toBuffer({ resolveWithObject: true });

    const { width, height } = info;
    const hogFeatures = [];
    const cellSize = 8;
    const numBins = 9;

    for (let y = 0; y < height - cellSize; y += cellSize) {
      for (let x = 0; x < width - cellSize; x += cellSize) {
        const cellHistogram = new Array(numBins).fill(0);

        for (let cy = 0; cy < cellSize; cy++) {
          for (let cx = 0; cx < cellSize; cx++) {
            const px = x + cx;
            const py = y + cy;

            if (px < width - 1 && py < height - 1) {
              const gx = data[py * width + (px + 1)] - data[py * width + (px - 1)];
              const gy = data[(py + 1) * width + px] - data[(py - 1) * width + px];
              
              const magnitude = Math.sqrt(gx * gx + gy * gy);
              let angle = Math.atan2(gy, gx) * (180 / Math.PI);
              if (angle < 0) angle += 180;

              const bin = Math.floor(angle / (180 / numBins)) % numBins;
              cellHistogram[bin] += magnitude;
            }
          }
        }

        // L2 normalization for the cell
        const norm = Math.sqrt(cellHistogram.reduce((sum, val) => sum + val * val, 0));
        hogFeatures.push(...cellHistogram.map(val => norm > 0 ? val / norm : 0));
      }
    }

    return hogFeatures.slice(0, 100); // Limit to first 100 features
  } catch (error) {
    console.error('Error extracting HOG features:', error);
    return new Array(100).fill(0);
  }
}

// Enhanced color histogram
async function extractColorHistogram(imageBuffer) {
  try {
    const { data, info } = await sharp(imageBuffer)
      .raw()
      .toBuffer({ resolveWithObject: true });

    const { width, height, channels } = info;
    const histograms = [[], [], []]; // RGB histograms
    const binSize = 32;

    // Initialize histograms
    for (let c = 0; c < channels; c++) {
      for (let i = 0; i < 8; i++) {
        histograms[c][i] = 0;
      }
    }

    // Build histograms
    for (let i = 0; i < data.length; i += channels) {
      for (let c = 0; c < channels; c++) {
        const bin = Math.floor(data[i + c] / binSize);
        if (bin >= 0 && bin < 8) {
          histograms[c][bin]++;
        }
      }
    }

    // Normalize and flatten
    const totalPixels = width * height;
    const flatHistogram = histograms.flat().map(val => val / totalPixels);

    // Add statistical moments
    const mean = flatHistogram.reduce((sum, val) => sum + val, 0) / flatHistogram.length;
    const std = Math.sqrt(flatHistogram.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / flatHistogram.length);

    return [...flatHistogram, mean, std];
  } catch (error) {
    console.error('Error extracting color histogram:', error);
    return new Array(26).fill(0);
  }
}

// Calculate similarity with feature weighting
function calculateFeatureSimilarity(vecA, vecB) {
  if (vecA.length !== vecB.length) return 0;

  // Feature type boundaries (adjust based on your feature extraction)
  const colorEnd = 9;        // Color moments: 9 features
  const textureEnd = 265;    // LBP: 256 features  
  const shapeEnd = 365;      // HOG: 100 features
  const histogramEnd = 391;  // Color histogram: 26 features

  const colorA = vecA.slice(0, colorEnd);
  const colorB = vecB.slice(0, colorEnd);
  const textureA = vecA.slice(colorEnd, textureEnd);
  const textureB = vecB.slice(colorEnd, textureEnd);
  const shapeA = vecA.slice(textureEnd, shapeEnd);
  const shapeB = vecB.slice(textureEnd, shapeEnd);
  const histogramA = vecA.slice(shapeEnd, histogramEnd);
  const histogramB = vecB.slice(shapeEnd, histogramEnd);

  // Calculate individual similarities
  const colorSim = cosineSimilarity(colorA, colorB);
  const textureSim = cosineSimilarity(textureA, textureB);
  const shapeSim = cosineSimilarity(shapeA, shapeB);
  const histogramSim = cosineSimilarity(histogramA, histogramB);

  console.log('Component similarities - Color:', colorSim.toFixed(3), 
              'Texture:', textureSim.toFixed(3), 
              'Shape:', shapeSim.toFixed(3),
              'Histogram:', histogramSim.toFixed(3));

  // Weighted combination (adjust weights based on importance)
  const finalSimilarity = (
    colorSim * 0.25 + 
    textureSim * 0.30 + 
    shapeSim * 0.30 + 
    histogramSim * 0.15
  );

  return Math.max(0, Math.min(1, finalSimilarity));
}

// Cosine similarity function
function cosineSimilarity(vecA, vecB) {
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  
  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  
  if (normA === 0 || normB === 0) {
    return 0;
  }
  
  const similarity = dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  return isNaN(similarity) ? 0 : similarity;
}

// Search for similar images
async function searchSimilarImages(queryEmbedding, topK = 5) {
  const connection = await pool.getConnection();
  
  try {
    const [rows] = await connection.execute('SELECT id, image_url, metadata, embedding FROM designs');
    const results = [];
    
    for (const item of rows) {
      try {
        const embedding = item.embedding;
        
        if (!embedding || !Array.isArray(embedding) || queryEmbedding.length !== embedding.length) {
          continue;
        }
        
        const similarity = calculateFeatureSimilarity(queryEmbedding, embedding);
        console.log(`Final similarity with ${item.id}:`, similarity.toFixed(3));
        var similar = parseFloat(similarity.toFixed(3));
        
        // ✅ only include if similarity ≥ 0.9
        if (similar >= 0.7) {
          results.push({
            uploadedImage:item.image_url,
            id: item.id,
            metadata: item.metadata,
            similarity: similar
          });
        }

        // results.push({
        //     uploadedImage:item.image_url,
        //     id: item.id,
        //     metadata: item.metadata,
        //     similarity: similar
        //   });

      } catch (error) {
        console.error('Error processing design:', item.id, error);
      }
    }
    
    return results
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, topK);
  } finally {
    connection.release();
  }
}


// Routes
app.post('/upload', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No image file provided' });
    }

    // ✅ save uploaded file into public/uploads
    
    const embedding = await extractAdvancedFeatures(req.file.buffer);
    const results = await searchSimilarImages(embedding);

    res.json({
      success: true,
      results: results          // ✅ already filtered by ≥ 90%
    });
  } catch (error) {
    console.error('Error processing upload:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

app.post('/index', upload.single('image'), async (req, res) => {
  const connection = await pool.getConnection();

  try {
    if (!req.file) return res.status(400).json({ error: 'No image file provided' });
    if (!req.body.id) return res.status(400).json({ error: 'Design ID is required' });

    const [existing] = await connection.execute('SELECT id FROM designs WHERE id = ?', [req.body.id]);
    if (existing.length > 0) return res.status(400).json({ error: 'Design ID already exists' });

    // ✅ save uploaded file into public/uploads
    const uploadsDir = path.join(__dirname, 'public', 'uploads');
    if (!fs.existsSync(uploadsDir)) fs.mkdirSync(uploadsDir, { recursive: true });
    const filename = Date.now() + '-' + req.file.originalname;
    const filePath = path.join(uploadsDir, filename);
    fs.writeFileSync(filePath, req.file.buffer);

    const imageUrl = `/uploads/${filename}`;

    const embedding = await extractAdvancedFeatures(req.file.buffer);
    let metadata = {};

    if (req.body.metadata) {
      try {
        metadata = JSON.parse(req.body.metadata);
      } catch (e) {
        console.error('Error parsing metadata:', e);
      }
    }

    await connection.execute(
      'INSERT INTO designs (id, image_url, metadata, embedding) VALUES (?, ?, ?, ?)',
      [req.body.id, imageUrl, JSON.stringify(metadata), JSON.stringify(embedding)]
    );

    res.json({
      success: true,
      message: 'Design indexed successfully',
      id: req.body.id,
      imageUrl: imageUrl
    });
  } catch (error) {
    console.error('Error indexing design:', error);
    res.status(500).json({ error: 'Internal server error' });
  } finally {
    connection.release();
  }
});


// Get all indexed designs
app.get('/designs', async (req, res) => {
  const connection = await pool.getConnection();
  try {
    const [rows] = await connection.execute(
      'SELECT id, image_url, metadata, created_at FROM designs ORDER BY created_at DESC'
    );
    res.json({ success: true, count: rows.length, designs: rows });
  } catch (error) {
    console.error('Error retrieving designs:', error);
    res.status(500).json({ error: 'Internal server error' });
  } finally {
    connection.release();
  }
});


// Delete a design
app.delete('/designs/:id', async (req, res) => {
  const connection = await pool.getConnection();
  
  try {
    const [result] = await connection.execute(
      'DELETE FROM designs WHERE id = ?',
      [req.params.id]
    );
    
    if (result.affectedRows === 0) {
      return res.status(404).json({ error: 'Design not found' });
    }
    
    res.json({
      success: true,
      message: 'Design deleted successfully'
    });
  } catch (error) {
    console.error('Error deleting design:', error);
    res.status(500).json({ error: 'Internal server error' });
  } finally {
    connection.release();
  }
});


// Initialize and start server
async function startServer() {
  try {
    await initializeDatabase();
    
    app.listen(PORT, () => {
      console.log(`Server running on http://localhost:${PORT}`);
      console.log('Using advanced feature extraction (no MobileNet)');
    });
  } catch (error) {
    console.error('Failed to start server:', error);
    process.exit(1);
  }
}

startServer();