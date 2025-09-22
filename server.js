const express = require('express');
const multer = require('multer');
const cors = require('cors');
const tf = require('@tensorflow/tfjs-node');
const mysql = require('mysql2/promise');
const sharp = require('sharp');
const path = require('path');
const fs = require('fs');
const crypto = require('crypto');

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
    fileSize: 10 * 1024 * 1024, // 10MB limit
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
  queueLimit: 0,
  typeCast: function (field, next) {
    if (field.type === 'JSON') {
      try {
        return JSON.parse(field.string());
      } catch (e) {
        return field.string();
      }
    }
    return next();
  }
};

// Create MySQL connection pool
const pool = mysql.createPool(dbConfig);

// Load MobileNet model
let model;
async function loadModel() {
  try {
    console.log('Loading MobileNet model...');
    model = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
    console.log('MobileNet model loaded successfully');
  } catch (error) {
    console.error('Error loading model:', error);
    // Fallback to local model if online loading fails
    try {
      console.log('Trying local model...');
      // You can download the model and serve it locally
      model = await tf.loadLayersModel('file://./model/model.json');
      console.log('Local model loaded successfully');
    } catch (localError) {
      console.error('Error loading local model:', localError);
      throw new Error('Could not load any model');
    }
  }
}

// Initialize database tables
async function initializeDatabase() {
  try {
    const connection = await pool.getConnection();
    
    // Create designs table if it doesn't exist
    await connection.execute(`
      CREATE TABLE IF NOT EXISTS designs (
        id INT AUTO_INCREMENT PRIMARY KEY,
        image_url VARCHAR(512),
        metadata JSON,
        embedding JSON,
        phash VARCHAR(64),
        product_type VARCHAR(100),
        brand VARCHAR(100),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX phash_idx (phash),
        INDEX product_type_idx (product_type),
        INDEX brand_idx (brand)
      )
    `);
    
    connection.release();
    console.log('Database initialized successfully');
  } catch (error) {
    console.error('Error initializing database:', error);
    throw error;
  }
}

// Enhanced image preprocessing for fashion items
async function preprocessImage(imageBuffer) {
  try {
    // Remove background and focus on the product
    const processed = await sharp(imageBuffer)
      .resize(224, 224, {
        fit: 'contain',
        background: { r: 255, g: 255, b: 255, alpha: 1 } // White background
      })
      .normalize()
      .sharpen()
      .jpeg()
      .toBuffer();

    return processed;
  } catch (error) {
    console.error('Error preprocessing image:', error);
    throw error;
  }
}

// Extract features using MobileNet
async function extractDeepFeatures(imageBuffer) {
  try {
    const processedBuffer = await preprocessImage(imageBuffer);
    
    // Decode image to tensor
    const imageTensor = tf.node.decodeImage(processedBuffer, 3);
    
    // Ensure the image has 3 channels (RGB)
    let finalTensor = imageTensor;
    if (imageTensor.shape[2] === 4) {
      finalTensor = imageTensor.slice([0, 0, 0], [224, 224, 3]);
    }
    
    // Normalize to [-1, 1]
    const normalized = finalTensor.toFloat().div(tf.scalar(127.5)).sub(tf.scalar(1));
    
    // Add batch dimension
    const batched = normalized.expandDims(0);
    
    // Get features from the model
    const predictions = model.predict(batched);
    const features = predictions.dataSync();
    
    // Clean up tensors
    tf.dispose([imageTensor, finalTensor, normalized, batched, predictions]);
    
    return Array.from(features);
  } catch (error) {
    console.error('Error extracting deep features:', error);
    throw error;
  }
}

// Perceptual Hash for exact duplicates
async function generatePerceptualHash(imageBuffer) {
  try {
    const resized = await sharp(imageBuffer)
      .resize(32, 32)
      .grayscale()
      .raw()
      .toBuffer({ resolveWithObject: true });

    const { data } = resized;
    let total = 0;
    
    for (let i = 0; i < data.length; i++) {
      total += data[i];
    }
    
    const average = total / data.length;
    let hash = '';
    
    for (let i = 0; i < data.length; i++) {
      hash += data[i] > average ? '1' : '0';
    }

    return crypto.createHash('md5').update(hash).digest('hex');
  } catch (error) {
    console.error('Error generating perceptual hash:', error);
    return null;
  }
}

// Enhanced similarity calculation for fashion items
function calculateSimilarity(vecA, vecB) {
  if (!vecA || !vecB || vecA.length !== vecB.length) return 0;
  
  // Use cosine similarity for deep learning features
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  
  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  
  if (normA === 0 || normB === 0) return 0;
  
  const similarity = dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  return Math.max(0, similarity);
}

// Search with metadata filtering
async function searchSimilarImages(queryEmbedding, queryPhash, filters = {}, topK = 10) {
  const connection = await pool.getConnection();
  
  try {
    let query = 'SELECT id, image_url, metadata, embedding, phash, product_type, brand FROM designs';
    const params = [];
    const conditions = [];
    
    // Add filters if provided
    if (filters.product_type) {
      conditions.push('product_type = ?');
      params.push(filters.product_type);
    }
    if (filters.brand) {
      conditions.push('brand = ?');
      params.push(filters.brand);
    }
    
    if (conditions.length > 0) {
      query += ' WHERE ' + conditions.join(' AND ');
    }
    
    const [rows] = await connection.execute(query, params);
    const results = [];
    
    for (const item of rows) {
      try {
        let embedding;
        try {
          embedding = typeof item.embedding === 'string' ? JSON.parse(item.embedding) : item.embedding;
        } catch (e) {
          console.error('Error parsing embedding for item:', item.id, e);
          continue;
        }
        
        if (!embedding || !Array.isArray(embedding)) {
          continue;
        }
        
        // Check for exact duplicates first
        if (queryPhash && item.phash) {
          const hammingDistance = calculateHammingDistance(queryPhash, item.phash);
          if (hammingDistance <= 3) { // Very close match
            results.push({
              id: item.id,
              image_url: item.image_url,
              metadata: item.metadata,
              similarity: 1.0 - (hammingDistance * 0.1),
              isExactMatch: hammingDistance === 0
            });
            continue;
          }
        }
        
        // Calculate deep learning similarity
        const similarity = calculateSimilarity(queryEmbedding, embedding);
        
        if (similarity >= 0.7) { // Adjust threshold as needed
          results.push({
            id: item.id,
            image_url: item.image_url,
            metadata: item.metadata,
            similarity: parseFloat(similarity.toFixed(3)),
            isExactMatch: false
          });
        }
        
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

function calculateHammingDistance(hash1, hash2) {
  if (!hash1 || !hash2 || hash1.length !== hash2.length) return Infinity;
  
  let distance = 0;
  for (let i = 0; i < hash1.length; i++) {
    if (hash1[i] !== hash2[i]) distance++;
  }
  return distance;
}

// Routes
app.post('/upload', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No image file provided' });
    }

    const [embedding, phash] = await Promise.all([
      extractDeepFeatures(req.file.buffer),
      generatePerceptualHash(req.file.buffer)
    ]);

    // Extract filters from query parameters
    const filters = {
      product_type: req.query.product_type,
      brand: req.query.brand
    };

    const results = await searchSimilarImages(embedding, phash, filters, 12);

    res.json({
      success: true,
      results: results,
      totalMatches: results.length
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

    const [embedding, phash] = await Promise.all([
      extractDeepFeatures(req.file.buffer),
      generatePerceptualHash(req.file.buffer)
    ]);

    // Save file
    const uploadsDir = path.join(__dirname, 'public', 'uploads');
    if (!fs.existsSync(uploadsDir)) fs.mkdirSync(uploadsDir, { recursive: true });
    
    const filename = `${Date.now()}-${crypto.randomBytes(8).toString('hex')}${path.extname(req.file.originalname)}`;
    const filePath = path.join(uploadsDir, filename);
    fs.writeFileSync(filePath, req.file.buffer);

    const imageUrl = `/uploads/${filename}`;
    let metadata = {};
    let product_type = '';
    let brand = '';

    if (req.body.metadata) {
      try {
        metadata = JSON.parse(req.body.metadata);
        product_type = metadata.product_type || '';
        brand = metadata.brand || '';
      } catch (e) {
        console.error('Error parsing metadata:', e);
      }
    }

    await connection.execute(
      'INSERT INTO designs (image_url, metadata, embedding, phash, product_type, brand) VALUES (?, ?, ?, ?, ?, ?)',
      [imageUrl, JSON.stringify(metadata), JSON.stringify(embedding), phash, product_type, brand]
    );

    res.json({
      success: true,
      message: 'Design indexed successfully',
      imageUrl: imageUrl
    });
  } catch (error) {
    console.error('Error indexing design:', error);
    res.status(500).json({ error: 'Internal server error' });
  } finally {
    connection.release();
  }
});

// Get all indexed designs with pagination - FIXED
app.get('/designs', async (req, res) => {
  const connection = await pool.getConnection();
  try {
    const page = Math.max(1, parseInt(req.query.page) || 1);
    const limit = Math.max(1, Math.min(100, parseInt(req.query.limit) || 20));
    const offset = (page - 1) * limit;

    // Use template literals for LIMIT and OFFSET to avoid parameter issues
    const [rows] = await connection.execute(
      `SELECT id, image_url, metadata, product_type, brand, created_at FROM designs ORDER BY created_at DESC LIMIT ${limit} OFFSET ${offset}`
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

// Health check endpoint
app.get('/health', async (req, res) => {
  try {
    const connection = await pool.getConnection();
    connection.release();
    
    res.json({ 
      status: 'healthy', 
      model: 'MobileNet v1', 
      timestamp: new Date().toISOString() 
    });
  } catch (error) {
    res.status(500).json({ status: 'unhealthy', error: error.message });
  }
});

// Initialize and start server
async function startServer() {
  try {
    await loadModel();
    await initializeDatabase();
    
    app.listen(PORT, () => {
      console.log(`Server running on http://localhost:${PORT}`);
      console.log('Using MobileNet deep learning model for image similarity');
    });
  } catch (error) {
    console.error('Failed to start server:', error);
    process.exit(1);
  }
}

startServer();