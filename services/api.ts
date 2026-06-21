const BASE_URL = "https://seallike-shalonda-nonsynchronous.ngrok-free.dev";

export const uploadImageToServer = async (imageUri: string) => {
  const formData = new FormData();

  formData.append("file", {
    uri: imageUri,
    name: "upload.jpg",
    type: "image/jpeg",
  } as any);

  const response = await fetch(`${BASE_URL}/predict`, {
    method: "POST",
    body: formData,
  });

  return await response.json();
};