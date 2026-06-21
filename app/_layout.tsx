import { Stack } from 'expo-router';
import theme from '../constants/theme';

export default function RootLayout() {
  return (
    <Stack
      screenOptions={{
        headerStyle: { backgroundColor: theme.bg },
        headerTintColor: theme.text,
        headerTitleStyle: { fontWeight: '700' },
        headerShadowVisible: false,
        contentStyle: { backgroundColor: theme.bg },
      }}
    >
      <Stack.Screen name="index" options={{ headerShown: false }} />
      <Stack.Screen name="upload" options={{ title: '업로드' }} />
      <Stack.Screen name="loading" options={{ headerShown: false }} />
      <Stack.Screen name="result" options={{ title: '분석 결과' }} />
      <Stack.Screen name="report" options={{ title: 'AI 리포트' }} />
    </Stack>
  );
}