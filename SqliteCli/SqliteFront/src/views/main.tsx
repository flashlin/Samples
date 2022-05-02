import { defineComponent, Fragment, reactive } from "vue";

export default defineComponent({
  setup(props, { expose }) {
    let state = reactive({
      result: ""
    });

    const clickFetchSend = () => {
      let userData = {
        id: 123,
        name: "flash",
      };
      fetch("http://example.com/movies.json", {
        method: "POST",
        headers: {
          "Accept": "application/json",
          "Content-Type": "application/json",
          //"Authorization": `Bearer ${token}`,
        },
        body: JSON.stringify(userData),
      }).then(function (response) {
        console.log("fetch resp", response);
        state.result = response as any;
      });
    };

    return () => (
      <div>
        Hello
        <button onClick={clickFetchSend}>Fetch POST</button>
         result = '{state.result}'
      </div>
    );
  },
});
