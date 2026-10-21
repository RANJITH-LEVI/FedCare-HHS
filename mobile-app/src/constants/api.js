import AsyncStorage from '@react-native-async-storage/async-storage';
import { Platform } from 'react-native';

const STORAGE_KEY = '@fedcare_api_url';

// Auto-detect best default URL:
// - Physical phone on Wi-Fi: host IP 10.185.148.248:8000
// - Android Emulator: 10.0.2.2:8000
// - Web / iOS Sim: localhost:8000
export const DEFAULT_HOST_IP = '10.185.148.248';

export const getDefaultApiUrl = () => {
  if (Platform.OS === 'web') {
    return 'http://localhost:8000';
  }
  return `http://${DEFAULT_HOST_IP}:8000`;
};

let cachedApiUrl = getDefaultApiUrl();

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
