const express = require('express');
const multer = require('multer');
const cors = require('cors');
const tf = require('@tensorflow/tfjs-node');
const mysql = require('mysql2/promise');
const sharp = require('sharp');
const path = require('path');
const fs = require('fs');
const crypto = require('crypto');
const axios = require('axios'); // Add axios for HTTP requests

const app = express();
const PORT = process.env.PORT || 9002;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static('public'));

// Configure multer for file uploads
const storage = multer.memoryStorage();
const upload = multer({ 
  storage: storage,
  limits: {
    fileSize: 10 * 1024 * 1024,
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

const pool = mysql.createPool(dbConfig);

// Recommendation API
const RECOMMENDED_API = "http://localhost:8000/api/ViewRecommendedProducts";

// Use Universal Sentence Encoder's image module or a more appropriate model
let model;
async function loadModel() {
  try {
    console.log('Loading image feature extraction model...');
    
    // Try to load a more suitable model for feature extraction
    // MobileNet is good but we need to use the right layer for features
    model = await tf.loadGraphModel('https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v2_100_224/feature_vector/3/default/1');
    console.log('MobileNet V2 feature extractor loaded successfully');
  } catch (error) {
    console.error('Error loading feature extraction model:', error);
    
    // Fallback to regular MobileNet
    try {
      model = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
      console.log('Standard MobileNet loaded as fallback');
    } catch (fallbackError) {
      console.error('Failed to load any model:', fallbackError);
      throw new Error('Could not load image model');
    }
  }
}

// Initialize database
async function initializeDatabase() {
  try {
    const connection = await pool.getConnection();
    await connection.execute(`
      CREATE TABLE IF NOT EXISTS designs (
        id INT AUTO_INCREMENT PRIMARY KEY,
        image_url VARCHAR(512),
        metadata JSON,
        embedding JSON,
        phash VARCHAR(64),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX phash_idx (phash)
      )
    `);
    connection.release();
    console.log('Database initialized');
  } catch (error) {
    console.error('Database initialization error:', error);
    throw error;
  }
}

// Enhanced preprocessing for better feature extraction
async function preprocessImage(imageBuffer) {
  try {
    const metadata = await sharp(imageBuffer).metadata();
    
    // Auto-orient and strip EXIF data
    let processed = sharp(imageBuffer)
      .rotate() // Auto-rotate based on EXIF
      .resize(224, 224, {
        fit: 'cover',
        position: 'center',
        withoutEnlargement: true
      })
      .normalize()
      .linear(1.1, 0) // Slight contrast enhancement
      .jpeg({ quality: 90 });

    return await processed.toBuffer();
  } catch (error) {
    console.error('Preprocessing error:', error);
    throw error;
  }
}

// Extract features with better normalization
async function extractDeepFeatures(imageBuffer) {
  try {
    const processedBuffer = await preprocessImage(imageBuffer);
    const tensor = tf.node.decodeImage(processedBuffer, 3);
    
    // Normalize to [0, 1] instead of [-1, 1] for better similarity
    const normalized = tensor.toFloat().div(tf.scalar(255));
    const batched = normalized.expandDims(0);
    
    let features;
    if (model instanceof tf.GraphModel) {
      // For feature vector models
      features = model.predict(batched);
    } else {
      // For classification models, use intermediate layers
      const layer = model.getLayer('conv_pw_13_relu'); // Use a deeper layer
      const intermediateModel = tf.model({
        inputs: model.inputs,
        outputs: layer.output
      });
      features = intermediateModel.predict(batched);
    }
    
    const featureArray = Array.from(features.dataSync());
    
    // Clean up
    tf.dispose([tensor, normalized, batched, features]);
    
    console.log(`Extracted ${featureArray.length} features`);
    return featureArray;
  } catch (error) {
    console.error('Feature extraction error:', error);
    
    // Fallback to traditional features if deep learning fails
    return await extractTraditionalFeatures(imageBuffer);
  }
}

// Traditional feature extraction as fallback
async function extractTraditionalFeatures(imageBuffer) {
  try {
    const [colorFeatures, textureFeatures, shapeFeatures] = await Promise.all([
      extractColorFeatures(imageBuffer),
      extractTextureFeatures(imageBuffer),
      extractShapeFeatures(imageBuffer)
    ]);
    
    return [...colorFeatures, ...textureFeatures, ...shapeFeatures];
  } catch (error) {
    console.error('Traditional feature extraction failed:', error);
    return new Array(512).fill(0); // Return empty features
  }
}

async function extractColorFeatures(imageBuffer) {
  const { data, info } = await sharp(imageBuffer)
    .resize(64, 64)
    .raw()
    .toBuffer({ resolveWithObject: true });
  
  const histograms = [[], [], []];
  for (let i = 0; i < data.length; i += 3) {
    for (let c = 0; c < 3; c++) {
      const bin = Math.floor(data[i + c] / 32);
      histograms[c][bin] = (histograms[c][bin] || 0) + 1;
    }
  }
  
  return histograms.flat().map(val => val / (64 * 64));
}

async function extractTextureFeatures(imageBuffer) {
  const { data, info } = await sharp(imageBuffer)
    .grayscale()
    .resize(32, 32)
    .raw()
    .toBuffer({ resolveWithObject: true });
  
  const features = [];
  for (let y = 1; y < 31; y++) {
    for (let x = 1; x < 31; x++) {
      const center = data[y * 32 + x];
      let pattern = 0;
      const neighbors = [
        data[(y-1)*32 + (x-1)], data[(y-1)*32 + x], data[(y-1)*32 + (x+1)],
        data[y*32 + (x-1)], data[y*32 + (x+1)],
        data[(y+1)*32 + (x-1)], data[(y+1)*32 + x], data[(y+1)*32 + (x+1)]
      ];
      
      neighbors.forEach((neighbor, idx) => {
        if (neighbor >= center) pattern |= (1 << idx);
      });
      features.push(pattern);
    }
  }
  
  return features.slice(0, 100); // Limit features
}

async function extractShapeFeatures(imageBuffer) {
  const { data, info } = await sharp(imageBuffer)
    .grayscale()
    .resize(64, 64)
    .raw()
    .toBuffer({ resolveWithObject: true });
  
  const edges = [];
  for (let y = 1; y < 63; y++) {
    for (let x = 1; x < 63; x++) {
      const gx = data[y*64 + (x+1)] - data[y*64 + (x-1)];
      const gy = data[(y+1)*64 + x] - data[(y-1)*64 + x];
      edges.push(Math.sqrt(gx*gx + gy*gy));
    }
  }
  
  return edges.slice(0, 100);
}

// Perceptual hash
async function generatePerceptualHash(imageBuffer) {
  try {
    const { data } = await sharp(imageBuffer)
      .resize(32, 32)
      .grayscale()
      .raw()
      .toBuffer({ resolveWithObject: true });
    
    let total = data.reduce((sum, val) => sum + val, 0);
    const avg = total / data.length;
    let hash = '';
    
    for (let val of data) {
      hash += val > avg ? '1' : '0';
    }
    
    return crypto.createHash('md5').update(hash).digest('hex');
  } catch (error) {
    console.error('Perceptual hash error:', error);
    return null;
  }
}

// Enhanced similarity calculation
function calculateSimilarity(vecA, vecB) {
  if (!vecA || !vecB || vecA.length !== vecB.length) return 0;
  
  // Use multiple similarity measures
  const cosineSim = cosineSimilarity(vecA, vecB);
  const euclideanSim = 1 / (1 + euclideanDistance(vecA, vecB));
  
  // Weighted combination
  return (cosineSim * 0.7 + euclideanSim * 0.3);
}

function cosineSimilarity(vecA, vecB) {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < vecA.length; i++) {
    dot += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  return normA && normB ? dot / (Math.sqrt(normA) * Math.sqrt(normB)) : 0;
}

function euclideanDistance(vecA, vecB) {
  let sum = 0;
  for (let i = 0; i < vecA.length; i++) {
    sum += Math.pow(vecA[i] - vecB[i], 2);
  }
  return Math.sqrt(sum);
}

// Search function with better debugging
async function searchSimilarImages(queryEmbedding, queryPhash, topK = 10) {
  const connection = await pool.getConnection();
  const results = [];
  
  try {
    const [rows] = await connection.execute(
      'SELECT id, image_url, metadata, embedding, phash FROM designs'
    );
    
    console.log(`Searching through ${rows.length} images...`);
    
    for (const item of rows) {
      try {
        let embedding;
        let metadata;
        try {
          embedding = typeof item.embedding === 'string' ? 
            JSON.parse(item.embedding) : item.embedding;
          metadata = typeof item.metadata === 'string' ? 
            JSON.parse(item.metadata) : item.metadata;
        } catch (e) {
          continue;
        }
        
        if (!embedding || !Array.isArray(embedding)) continue;
        
        // Check perceptual hash first
        if (queryPhash && item.phash) {
          const hammingDist = calculateHammingDistance(queryPhash, item.phash);
          if (hammingDist <= 5) {
            results.push({
              id: item.id,
              image_url: item.image_url,
              similarity: 1.0 - (hammingDist * 0.1),
              isExactMatch: hammingDist === 0,
              metadata: metadata
            });
            continue;
          }
        }
        
        // Calculate feature similarity
        const similarity = calculateSimilarity(queryEmbedding, embedding);
        
        console.log(`Similarity with image ${item.id}: ${similarity.toFixed(3)}`);
        
        if (similarity >= 0.2) { // Lower threshold for better recall
          results.push({
            id: item.id,
            image_url: item.image_url,
            similarity: parseFloat(similarity.toFixed(3)),
            isExactMatch: false,
            metadata: metadata
          });
        }
      } catch (error) {
        console.error('Error processing item:', error);
      }
    }
    
    return results.sort((a, b) => b.similarity - a.similarity).slice(0, topK);
  } finally {
    connection.release();
  }
}

// NEW: Fetch recommended products based on product IDs
async function fetchRecommendedProducts(productIds) {
  try {
    if (!productIds || productIds.length === 0) {
      console.log('No product IDs provided for recommendations');
      return [];
    }

    console.log(`Fetching recommendations for product IDs: ${productIds.join(', ')}`);
    
    const response = await axios.post(RECOMMENDED_API, {
      productIds: productIds
    }, {
      timeout: 10000 // 10 second timeout
    });

    console.log(`[DEBUG] Recommended products fetched successfully. Count: ${response.data.length}`);
    return response.data || [];
  } catch (error) {
    console.error('Error fetching recommended products:', error.message);
    if (error.response) {
      console.error('Response status:', error.response.status);
      console.error('Response data:', error.response.data);
    }
    return [];
  }
}

function calculateHammingDistance(hash1, hash2) {
  if (!hash1 || !hash2 || hash1.length !== hash2.length) return Infinity;
  let dist = 0;
  for (let i = 0; i < hash1.length; i++) {
    if (hash1[i] !== hash2[i]) dist++;
  }
  return dist;
}

// Routes
app.post('/upload', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: 'No image provided' });
    
    console.log('Processing upload...');
    const [embedding, phash] = await Promise.all([
      extractDeepFeatures(req.file.buffer),
      generatePerceptualHash(req.file.buffer)
    ]);
    
    console.log('Searching for similar images...');
    const similarResults = await searchSimilarImages(embedding, phash, 20);
    
    // Extract product IDs from similar results for recommendations
    const productIds = similarResults
      .filter(result => result.metadata && result.metadata.productId)
      .map(result => result.metadata.productId)
      .slice(0, 5); // Use top 5 for recommendations
    
    console.log(`Extracted product IDs for recommendations: ${productIds}`);
    
    // Fetch recommended products
    let recommendedProducts = [];
    if (productIds.length > 0) {
      recommendedProducts = await fetchRecommendedProducts(productIds);
      console.log(`Found ${recommendedProducts.length} recommended products`);
    }
    
    res.json({
      success: true,
      results: similarResults,
      recommendedProducts: recommendedProducts, // Add recommended products to response
      totalMatches: similarResults.length,
      totalRecommendations: recommendedProducts.length,
      featuresLength: embedding.length
    });
  } catch (error) {
    console.error('Upload error:', error);
    res.status(500).json({ error: error.message });
  }
});

app.post('/index', upload.single('image'), async (req, res) => {
  const connection = await pool.getConnection();
  
  try {
    if (!req.file) return res.status(400).json({ error: 'No image provided' });
    
    const [embedding, phash] = await Promise.all([
      extractDeepFeatures(req.file.buffer),
      generatePerceptualHash(req.file.buffer)
    ]);
    
    // Save file
    const uploadsDir = path.join(__dirname, 'public', 'uploads');
    if (!fs.existsSync(uploadsDir)) fs.mkdirSync(uploadsDir, { recursive: true });
    
    const filename = `${Date.now()}-${Math.random().toString(36).substr(2, 9)}${path.extname(req.file.originalname)}`;
    const filePath = path.join(uploadsDir, filename);
    fs.writeFileSync(filePath, req.file.buffer);
    
    const imageUrl = `/uploads/${filename}`;
    const metadata = req.body.metadata ? JSON.parse(req.body.metadata) : {};
    
    await connection.execute(
      'INSERT INTO designs (image_url, metadata, embedding, phash) VALUES (?, ?, ?, ?)',
      [imageUrl, JSON.stringify(metadata), JSON.stringify(embedding), phash]
    );
    
    res.json({
      success: true,
      message: 'Image indexed successfully',
      imageUrl: imageUrl,
      featuresLength: embedding.length
    });
  } catch (error) {
    console.error('Index error:', error);
    res.status(500).json({ error: error.message });
  } finally {
    connection.release();
  }
});

// Get all designs
app.get('/designs', async (req, res) => {
  const connection = await pool.getConnection();
  try {
    const page = Math.max(1, parseInt(req.query.page) || 1);
    const limit = Math.max(1, Math.min(100, parseInt(req.query.limit) || 20));
    const offset = (page - 1) * limit;
    
    const [rows] = await connection.execute(
      `SELECT id, image_url, metadata, created_at FROM designs ORDER BY created_at DESC LIMIT ${limit} OFFSET ${offset}`
    );
    
    const [countRows] = await connection.execute('SELECT COUNT(*) as total FROM designs');
    const total = countRows[0].total;
    
    res.json({
      success: true,
      count: rows.length,
      total: total,
      page: page,
      pages: Math.ceil(total / limit),
      designs: rows
    });
  } catch (error) {
    console.error('Error retrieving designs:', error);
    res.status(500).json({ error: error.message });
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

// NEW: Direct recommendations endpoint (optional)
app.post('/recommendations', async (req, res) => {
  try {
    const { productIds } = req.body;
    
    if (!productIds || !Array.isArray(productIds)) {
      return res.status(400).json({ error: 'productIds array is required' });
    }
    
    const recommendedProducts = await fetchRecommendedProducts(productIds);
    
    res.json({
      success: true,
      recommendedProducts: recommendedProducts,
      count: recommendedProducts.length
    });
  } catch (error) {
    console.error('Recommendations error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Initialize server
async function startServer() {
  try {
    await loadModel();
    await initializeDatabase();
    
    app.listen(PORT, () => {
      console.log(`Server running on http://localhost:${PORT}`);
      console.log('Using enhanced image similarity system with product recommendations');
    });
  } catch (error) {
    console.error('Failed to start server:', error);
    process.exit(1);
  }
}

startServer();