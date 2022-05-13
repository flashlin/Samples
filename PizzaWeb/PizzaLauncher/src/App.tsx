import { defineComponent, ref } from "vue";

function* getFileChunks(file: File) {
  const chunkSize = 1024 * 100;
  //const sliceBlob = data.slice(0, 10);
  for (let start = 0; start < file.size; start += chunkSize) {
    const chunk = file.slice(start, start + chunkSize);
    yield chunk;
  }
}

export default defineComponent({
  props: {},
  setup(props) {
    const state = ref({
      imageId: 1,
    });

    const onClick = async (e: Event) => {
      const target = e.target as HTMLInputElement;
      const fileList = target.files;
      if (fileList == null) {
        return;
      }
      let isFirst = true;
      for (let chunk of getFileChunks(fileList[0])) {
        //let chunkBuffer = await chunk.arrayBuffer();

        let formData = new FormData();
        formData.append("imageId", state.value.imageId.toString());
        formData.append("image", chunk, "imageName");
        formData.append("isFirst", isFirst.toString());

        await fetch(`/api/StoreShelves/SaveBlob`, {
         //  headers: {
         //    "Content-Type": "multipart/form-data",
         //    //    //"Content-Type": "application/json",
         //    //    //Accept: "application/json",
         //    //    //"Authorization": `Bearer ${token}`,
         //  },
          method: "POST",
          body: formData,
        });
        isFirst = false;
      }
    };

    return () => (
      <div>
        <span>Hello world!</span>
        <input type="file" accept="image/*" onChange={onClick} />
      </div>
    );
  },
});
