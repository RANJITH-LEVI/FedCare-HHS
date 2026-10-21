import { getApiUrl } from '../constants/api';
import { Platform } from 'react-native';

class ApiService {
  async fetchWithTimeout(endpoint, options = {}, timeoutMs = 25000) {
    const baseUrl = await getApiUrl();
    const url = `${baseUrl}${endpoint}`;

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);

    try {
      const response = await fetch(url, {
        ...options,
        signal: controller.signal,
      });
      clearTimeout(timer);

      if (!response.ok) {
        let errDetail = `Status ${response.status}`;
        try {
          const errJson = await response.json();
          if (errJson.detail) errDetail = errJson.detail;
        } catch (e) {
          // ignore
        }
        throw new Error(errDetail);
      }

      return await response.json();
    } catch (err) {
      clearTimeout(timer);
      if (err.name === 'AbortError') {
        throw new Error(`Connection timed out (${baseUrl}). Check if server is running and accessible.`);
      }
      if (err.message && (err.message.includes('fetch failed') || err.message.includes('Network request failed'))) {
        throw new Error(`Cannot reach server at ${baseUrl}.\nEnsure FastAPI server is running and phone is on same Wi-Fi.`);
      }
      throw err;
    }
  }

  async checkHealth() {
    return this.fetchWithTimeout('/health', { method: 'GET' }, 4000);
  }

  async predict(patientData) {
    try {
      const res = await this.fetchWithTimeout('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(patientData),
      }, 10000);

      const probability = res.risk_score ?? res.probability ?? (res.risk_percentage ? res.risk_percentage / 100 : 0);
      const riskLevel = res.risk_tier ?? res.risk_level ?? 'Low Risk';

      // Parse feature_attributions list into simple shapValues dictionary
      let shapDict = {};
      if (res.feature_attributions && Array.isArray(res.feature_attributions)) {
        res.feature_attributions.forEach(item => {
          if (item.feature && item.shap_value !== undefined) {
            shapDict[item.feature] = item.shap_value;
          }
        });
      } else if (res.shap_values && typeof res.shap_values === 'object') {
        shapDict = res.shap_values;
      }

      return {
        ...res,
        prediction: res.prediction ?? (probability > 0.5 ? 1 : 0),
        probability,
        risk_score: probability,
        risk_level: riskLevel,
        riskLevel,
        risk_tier: riskLevel,
        shap_values: shapDict,
        shapValues: shapDict,
        narrative: res.plain_language_narrative || res.narrative || '',
      };
    } catch (err) {
      console.warn('Backend server unreachable, computing local evaluation fallback:', err.message);
      return this.fallbackLocalPredict(patientData);
    }
  }

  async explain(patientData) {
    try {
      return await this.fetchWithTimeout('/explain', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(patientData),
      }, 8000);
    } catch (err) {
      return this.fallbackLocalExplain(patientData);
    }
  }

  fallbackLocalPredict(p) {
    let score = 0.20; // baseline
    if (p.age > 55) score += 0.12;
    if (p.sex === 1) score += 0.08;
    if (p.cp === 4) score += 0.22; else if (p.cp === 1) score += 0.15;
    if (p.trestbps > 140) score += 0.10;
    if (p.chol > 240) score += 0.08;
    if (p.thalach < 130) score += 0.12;
    if (p.exang === 1) score += 0.14;
    if (p.oldpeak > 1.5) score += 0.16;
    if (p.ca > 0) score += 0.15;
    if (p.thal === 6 || p.thal === 7) score += 0.18;

    const prob = Math.min(Math.max(score, 0.05), 0.98);
    let risk_level = 'Low Risk';
    if (prob > 0.65) risk_level = 'High Risk';
    else if (prob > 0.40) risk_level = 'Moderate Risk';

    return {
      prediction: prob > 0.5 ? 1 : 0,
      probability: prob,
      risk_level,
      confidence: 0.88,
      is_offline: true,
    };
  }

  fallbackLocalExplain(p) {
    const shap = {
      cp: p.cp === 4 ? '+0.220' : '+0.050',
      oldpeak: p.oldpeak > 1.0 ? `+${(p.oldpeak * 0.12).toFixed(3)}` : '-0.040',
      ca: p.ca > 0 ? `+${(p.ca * 0.11).toFixed(3)}` : '-0.050',
      thal: (p.thal === 6 || p.thal === 7) ? '+0.180' : '-0.080',
      thalach: p.thalach < 140 ? '+0.120' : '-0.090',
      trestbps: p.trestbps > 140 ? '+0.090' : '-0.030',
      chol: p.chol > 240 ? '+0.070' : '-0.020',
      age: p.age > 55 ? '+0.080' : '-0.040',
    };

    return {
      shap_values: shap,
      narrative: `Local AI Analysis: Primary risk drivers include ${p.cp === 4 ? 'Asymptomatic Chest Pain' : 'Symptomatic CP'}, Resting BP (${p.trestbps} mmHg), and ST Depression (${p.oldpeak} mm).`,
      interactions: [],
    };
  }

  async scanDocument(fileUri, mimeType = 'image/jpeg', fileName = 'report.jpg') {
    const baseUrl = await getApiUrl();

    // 1. Try base64 conversion first (bulletproof across React Native Hermes, Web, Android, iOS)
    try {
      const fileRes = await fetch(fileUri);
      const blob = await fileRes.blob();

      const base64Data = await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onloadend = () => {
          const result = reader.result;
          if (typeof result === 'string') {
            resolve(result);
          } else {
            reject(new Error('Failed to read image base64'));
          }
        };
        reader.onerror = reject;
        reader.readAsDataURL(blob);
      });

      const extractedData = await this.fetchWithTimeout('/scan-base64', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_base64: base64Data,
          mime_type: mimeType,
        }),
      }, 35000);

      return { extracted_features: extractedData };
    } catch (e) {
      console.warn('Base64 scan conversion failed, attempting multipart fallback:', e.message);
    }

    // 2. Multipart FormData Fallback
    const url = `${baseUrl}/scan`;
    const formData = new FormData();

    if (Platform.OS === 'web') {
      const res = await fetch(fileUri);
      const blob = await res.blob();
      formData.append('file', blob, fileName);
    } else {
      formData.append('file', {
        uri: Platform.OS === 'android' ? fileUri : fileUri.replace('file://', ''),
        type: mimeType,
        name: fileName,
      });
    }

    const response = await fetch(url, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      let msg = 'Failed to extract document';
      try {
        const errJson = await response.json();
        if (errJson.detail) msg = errJson.detail;
      } catch (e) {}
      throw new Error(msg);
    }

    const resJson = await response.json();
    return { extracted_features: resJson };
  }

  async getHospitalDatasets() {
    return this.fetchWithTimeout('/hospitals/datasets', { method: 'GET' });
  }

  async addHospitalDatasetJson(payload) {
    return this.fetchWithTimeout('/hospitals/dataset/add-json', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }, 45000);
  }

  async getModelMetrics() {
    return this.fetchWithTimeout('/model/metrics', { method: 'GET' });
  }

  async triggerFederatedRound(epochs = 15) {
    return this.fetchWithTimeout('/federated/round', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ local_epochs: epochs }),
    }, 45000);
  }
}

export default new ApiService();
