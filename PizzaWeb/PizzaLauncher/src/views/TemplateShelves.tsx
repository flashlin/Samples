import { defineComponent, onMounted, reactive, ref } from "vue";

export default defineComponent({
	props: {},
	setup(props) {
		return () => (
			<div>
            Template Editor
				<span>Hello world!</span>
			</div>
		);
	},
});
