import AsyncStorage from '@react-native-async-storage/async-storage';

const HISTORY_KEY = '@fedcare_patient_history';

class StorageService {
  async getAssessments() {
    try {
      const data = await AsyncStorage.getItem(HISTORY_KEY);
      return data ? JSON.parse(data) : [];
    } catch (e) {
      console.error('Failed to load history', e);
      return [];
    }
  }

  async saveAssessment(assessment) {
    try {
      const current = await this.getAssessments();
      const newRecord = {
        id: Date.now().toString(),
        timestamp: new Date().toISOString(),
        ...assessment,
      };
      const updated = [newRecord, ...current].slice(0, 100); // Keep latest 100
      await AsyncStorage.setItem(HISTORY_KEY, JSON.stringify(updated));
      return newRecord;
    } catch (e) {
      console.error('Failed to save assessment', e);
      throw e;
    }
  }

  async deleteAssessment(id) {
    try {
      const current = await this.getAssessments();
      const updated = current.filter(item => item.id !== id);
      await AsyncStorage.setItem(HISTORY_KEY, JSON.stringify(updated));
      return updated;
    } catch (e) {
      console.error('Failed to delete assessment', e);
      throw e;
    }
  }

  async clearAll() {
    try {
      await AsyncStorage.removeItem(HISTORY_KEY);
      return [];
    } catch (e) {
      console.error('Failed to clear history', e);
      throw e;
    }
  }
}

export default new StorageService();
