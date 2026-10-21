import AsyncStorage from '@react-native-async-storage/async-storage';
import { Platform } from 'react-native';

const STORAGE_KEY = '@fedcare_api_url';

// Fixed Production Cloud Backend URL (Render.com live server)
export const DEFAULT_ONLINE_URL = 'https://fedcare-backend.onrender.com';
export const DEFAULT_HOST_IP = '10.185.148.248';

export const getDefaultApiUrl = () => {
  return DEFAULT_ONLINE_URL;
};

let cachedApiUrl = DEFAULT_ONLINE_URL;

export const getApiUrl = async () => {
  try {
    const saved = await AsyncStorage.getItem(STORAGE_KEY);
    if (saved) {
      cachedApiUrl = saved;
      return saved;
    }
  } catch (e) {
    // fallback
  }
  return cachedApiUrl;
};

export const setApiUrl = async (newUrl) => {
  let cleaned = newUrl.trim();
  if (cleaned.endsWith('/')) {
    cleaned = cleaned.slice(0, -1);
  }
  cachedApiUrl = cleaned;
  await AsyncStorage.setItem(STORAGE_KEY, cleaned);
  return cleaned;
};
