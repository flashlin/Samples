import { defineComponent, onMounted, reactive, ref } from "vue";

function* getFileChunks(file: File) {
	const chunkSize = 1024 * 100;
	for (let start = 0; start < file.size; start += chunkSize) {
		const chunk = file.slice(start, start + chunkSize);
		yield chunk;
	}
}

export interface StoreShelves {
	id: number;
	title: string;
	content: string;
	imageName: string;
}

async function getStoreShelves(): Promise<StoreShelves[]> {
	let resp = await fetch(`/api/StoreShelves/GetAll`);
	return resp.json();
}

export default defineComponent({
	props: {},
	setup(props) {
		const state = reactive({
			items: [] as StoreShelves[]
		});

		onMounted(async () => {
			state.items = await getStoreShelves();
		});

		const onClickImage = async (e: Event, imageId: number) => {
			const target = e.target as HTMLInputElement;
			const fileList = target.files;
			if (fileList == null) {
				return;
			}
			let isFirst = true;
			for (let chunk of getFileChunks(fileList[0])) {
				let formData = new FormData();
				formData.append("imageId", imageId.toString());
				formData.append("image", chunk, "imageName");
				formData.append("isFirst", isFirst.toString());

				await fetch(`/api/StoreShelves/SaveBlob`, {
					method: "POST",
					body: formData,
				});
				isFirst = false;
			}
		};

		const onClickUpdate = (imageId: number) => {
			let item = state.items.find((x) => x.id === imageId);
			console.log(item);
		};

		return () => (
			<div class="container">
				<span>Hello world!</span>

				<div class="row">
					{state.items.map((item: StoreShelves) =>
						<div class="card" style="width: 18rem;">
							<label>
								<input type="file" accept="image/*"
									style="display:none"
									onChange={(e) => onClickImage(e, item.id)} />
								<img src={`/images/img${item.id}.jpg`} class="card-img-top" alt="img" />
							</label>
							<div class="card-body">
								<h5 class="card-title">
									<input type="text" v-model={item.title} />
								</h5>
								<p class="card-text">
									<textarea v-model={item.content} rows="5" cols="33" />
								</p>
								<button class="btn btn-primary" onClick={e => onClickUpdate(item.id)}>Update</button>
							</div>
						</div>
					)}
				</div>

			</div>
		);
	},
});
