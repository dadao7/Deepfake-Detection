import { Ionicons } from '@expo/vector-icons';
import * as ImagePicker from 'expo-image-picker';
import { Stack, router } from 'expo-router';
import { useState } from 'react';
import {
  Alert,
  Image,
  Modal,
  Pressable,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import theme from '../constants/theme';
import { uploadImageToServer } from '../services/api';

export default function UploadScreen() {
  const [imageUri, setImageUri] = useState<string | null>(null);
  const [guideVisible, setGuideVisible] = useState(false);

  const pickImage = async () => {
    const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();

    if (!permission.granted) {
      Alert.alert(
        '권한 필요',
        '이미지를 선택하려면 사진 접근 권한이 필요합니다.'
      );
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      quality: 1,
    });

    if (!result.canceled) {
      setImageUri(result.assets[0].uri);
    }
  };

const startAnalysis = async () => {
  if (!imageUri) {
    Alert.alert('이미지 필요', '분석할 이미지를 먼저 선택해주세요.');
    return;
  }

  try {
    const response = await uploadImageToServer(imageUri);

    router.push({
      pathname: '/loading',
      params: {
        imageUri,
        jobId: response.job_id,
      },
    });

  } catch (error) {
    console.log(error);
    Alert.alert('오류', '서버 연결 또는 분석 중 문제가 발생했습니다.');
  }
};

  return (
    <>
      <Stack.Screen
        options={{
          title: '업로드',
          headerShadowVisible: false,
          headerRight: () => (
            <Pressable
              style={styles.headerGuideBtn}
              onPress={() => setGuideVisible(true)}
            >
              <Ionicons
                name="information-circle-outline"
                size={20}
                color={theme.accent}
              />
              <Text style={styles.headerGuideText}>Guide</Text>
            </Pressable>
          ),
        }}
      />

      <View style={styles.container}>
        <Text style={styles.title}>분석할 이미지 업로드</Text>

        <Text style={styles.desc}>
          얼굴이 선명하게 보이는 이미지를 선택하면 더 정확한 분석이
          가능합니다.
        </Text>

        <Pressable style={styles.uploadBox} onPress={pickImage}>
          {imageUri ? (
            <Image source={{ uri: imageUri }} style={styles.preview} />
          ) : (
            <>
              <Ionicons
                name="cloud-upload-outline"
                size={60}
                color={theme.accent}
              />
              <Text style={styles.uploadText}>탭하여 이미지 선택</Text>
            </>
          )}
        </Pressable>

        <View style={styles.infoBox}>
          <Text style={styles.infoText}>• JPG, PNG 이미지 지원</Text>
          <Text style={styles.infoText}>
            • 얼굴 영역이 클수록 분석 정확도 향상
          </Text>
          <Text style={styles.infoText}>
            • 분석 결과는 앱 화면에서 확인 가능
          </Text>
        </View>

        <Pressable style={styles.btn} onPress={startAnalysis}>
          <Text style={styles.btnText}>AI 분석 시작</Text>
        </Pressable>

        {/* Guide Modal */}
        <Modal
          visible={guideVisible}
          transparent
          animationType="fade"
          onRequestClose={() => setGuideVisible(false)}
        >
          <View style={styles.modalOverlay}>
            <View style={styles.modalBox}>
              {/* Header */}
              <View style={styles.modalHeader}>
                <Text style={styles.modalTitle}>업로드 가이드</Text>

                <Pressable onPress={() => setGuideVisible(false)}>
                  <Ionicons
                    name="close"
                    size={26}
                    color={theme.text}
                  />
                </Pressable>
              </View>

              {/* Description */}
              <Text style={styles.modalDesc}>
                정확한 분석을 위해 아래와 같은 이미지를
                권장합니다.
              </Text>

              {/* Good / Bad Example */}
              <View style={styles.exampleRow}>
                {/* Good */}
                <View style={styles.exampleCard}>
                  <Text style={styles.goodTitle}>✓ 좋은 예시</Text>

                  <View style={styles.goodImageBox}>
                    <Image
                      source={require('../assets/images/good1.png')}
                      style={styles.guideImage}
                      resizeMode="cover"
                    />
                  </View>

                  <Text style={styles.exampleText}>
                     정면 이미지
                  </Text>

                  <View style={styles.goodImageBox}>
                    <Image
                      source={require('../assets/images/good2.png')}
                      style={styles.guideImage}
                      resizeMode="cover"
                    />
                  </View>
                  <Text style={styles.exampleText}>
                    밝고 선명한 이미지
                  </Text>
                </View>

                {/* Bad */}
                <View style={styles.exampleCard}>
                  <Text style={styles.badTitle}>✕ 나쁜 예시</Text>

                  <View style={styles.badImageBox}>
                    <Image
                      source={require('../assets/images/bad1.png')}
                      style={styles.guideImage}
                      resizeMode="cover"
                    />
                  </View>

                  <Text style={styles.exampleText}>
                    어두운 이미지
                  </Text>

                  <View style={styles.badImageBox}>
                    <Image
                      source={require('../assets/images/bad2.png')}
                      style={styles.guideImage}
                      resizeMode="cover"
                    />
                  </View>

                  <Text style={styles.exampleText}>
                    얼굴이 가려진 이미지
                  </Text>
                </View>
              </View>

              {/* Note */}
              <Text style={styles.guideNote}>
                ⓘ 얼굴이 선명하고 크게 보일수록 분석 정확도가 향상됩니다.
              </Text>

              <Pressable
                style={styles.closeBtn}
                onPress={() => setGuideVisible(false)}
              >
                <Text style={styles.closeBtnText}>확인</Text>
              </Pressable>
            </View>
          </View>
        </Modal>
      </View>
    </>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 22,
    backgroundColor: theme.bg,
  },

  headerGuideBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },

  headerGuideText: {
    color: theme.accent,
    fontSize: 14,
    fontWeight: '700',
  },

  title: {
    color: theme.text,
    fontSize: 26,
    fontWeight: '800',
    marginTop: 20,
  },

  desc: {
    color: theme.subText,
    fontSize: 14,
    lineHeight: 22,
    marginTop: 10,
    marginBottom: 24,
  },

  uploadBox: {
    width: '100%',
    height: 330,
    borderRadius: 26,
    backgroundColor: theme.card,
    borderWidth: 1,
    borderColor: theme.border,
    alignItems: 'center',
    justifyContent: 'center',
    overflow: 'hidden',
  },

  preview: {
    width: '100%',
    height: '100%',
  },

  uploadText: {
    color: theme.text,
    fontSize: 16,
    fontWeight: '600',
    marginTop: 16,
  },

  infoBox: {
    backgroundColor: theme.card2,
    borderRadius: 18,
    padding: 18,
    marginTop: 24,
    borderWidth: 1,
    borderColor: theme.border,
  },

  infoText: {
    color: theme.subText,
    fontSize: 14,
    marginBottom: 8,
  },

  btn: {
    backgroundColor: theme.primary,
    paddingVertical: 18,
    borderRadius: 18,
    alignItems: 'center',
    marginTop: 'auto',
    marginBottom: 20,
  },

  btnText: {
    color: theme.text,
    fontSize: 17,
    fontWeight: '700',
  },

  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.6)',
    justifyContent: 'center',
    padding: 20,
  },

  modalBox: {
    backgroundColor: theme.card,
    borderRadius: 24,
    padding: 20,
    borderWidth: 1,
    borderColor: theme.border,
  },

  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },

  modalTitle: {
    color: theme.text,
    fontSize: 24,
    fontWeight: '800',
  },

  modalDesc: {
    color: theme.subText,
    fontSize: 14,
    lineHeight: 21,
    marginTop: 12,
    marginBottom: 18,
  },

  exampleRow: {
    flexDirection: 'row',
    gap: 12,
  },

  exampleCard: {
    flex: 1,
    borderWidth: 1,
    borderColor: theme.border,
    borderRadius: 18,
    padding: 12,
    backgroundColor: theme.card2,
  },

  goodTitle: {
    color: '#6BE675',
    fontSize: 16,
    fontWeight: '800',
    textAlign: 'center',
    marginBottom: 12,
  },

  badTitle: {
    color: '#FF6B6B',
    fontSize: 16,
    fontWeight: '800',
    textAlign: 'center',
    marginBottom: 12,
  },

  goodImageBox: {
    height: 90,
    borderRadius: 14,
    backgroundColor: 'rgba(107, 230, 117, 0.12)',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 8,
  },

  badImageBox: {
    height: 90,
    borderRadius: 14,
    backgroundColor: 'rgba(255, 107, 107, 0.12)',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 8,
  },

  exampleText: {
    color: theme.subText,
    fontSize: 12,
    lineHeight: 17,
    textAlign: 'center',
    marginBottom: 14,
  },

  guideNote: {
    color: theme.subText,
    fontSize: 13,
    lineHeight: 20,
    textAlign: 'center',
    marginTop: 18,
  },

  closeBtn: {
    backgroundColor: theme.primary,
    marginTop: 20,
    paddingVertical: 15,
    borderRadius: 16,
    alignItems: 'center',
  },

  closeBtnText: {
    color: theme.text,
    fontSize: 16,
    fontWeight: '700',
  },
  guideImage: {
  width: '100%',
  height: '100%',
  borderRadius: 14,
},
});