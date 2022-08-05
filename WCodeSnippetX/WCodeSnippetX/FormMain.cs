using System.Data;

namespace WCodeSnippetX
{
	public partial class FormMain : Form
	{
		DataTable _table = new DataTable();

		public FormMain()
		{
			InitializeComponent();
		}

		void Init()
		{
			_table.Columns.Add("idx");
			_table.Columns.Add("content");

			DataRow row = _table.NewRow();
			row[0] = "1";
			row[1] = "Datagridview and richtextbox for bold substring in C#";
			_table.Rows.Add(row);
			_table.AcceptChanges();

			//this.dataGridView1.AutoGenerateColumns = false;
			//this.dataGridView1.DataSource = _table;
		}

		private void dataGridView_CellContentClick(object sender, DataGridViewCellEventArgs e)
		{

		}
	}
}