using System.Data;
using WCodeSnippetX.Models;

namespace WCodeSnippetX
{
	public partial class FormMain : Form
	{
		readonly DataTable _table = new DataTable();
		BindingSource _bindingSource = new BindingSource();
		readonly DataGridView _dataGridView = new DataGridView();

		public FormMain()
		{
			InitializeComponent();
			Init();
		}

		void Init()
		{
			_table.Columns.Add(new DataColumn("idx", typeof(int)));
			_table.Columns.Add(new DataColumn("context", typeof(string)));

			DataRow row = _table.NewRow();
			row[0] = "1";
			_table.Rows.Add(row);
			_table.AcceptChanges();

			_bindingSource.Add(new CodeSnippet { Id = 1, Content = "Datagridview and richtextbox for bold substring in C#" });
			_bindingSource.Add(new CodeSnippet { Id = 2, Content = "Sample htextbox for bold substring in C#" });
			_bindingSource.Add(new CodeSnippet { Id = 3, Content = "public class { \r\n public string Name; }" });
			//_bindingSource.DataSource = _table;

			_dataGridView.Columns.Add(new DataGridViewTextBoxColumn()
			{
				Name = "index",
				DataPropertyName = "Id",
				ReadOnly = true,
				Width = 3 * 12,
			});
			_dataGridView.Columns.Add(new DataGridViewTextBoxColumn()
			{
				Name = "Code Content",
				DataPropertyName = "content",
				ReadOnly = true,
			});

			_dataGridView.RowTemplate.Height = 12 * 5;
			_dataGridView.AutoGenerateColumns = false;
			_dataGridView.DataSource = _bindingSource;
			_dataGridView.Height = ClientSize.Height - buttonSearch.Height - 6;
			this.Controls.Add(_dataGridView);
		}

		private void FormMain_ResizeEnd(object sender, EventArgs e)
		{
			_dataGridView.Height = ClientSize.Height - buttonSearch.Height - 6;
			buttonSearch.Left = textBoxSearch.Right + 3;
		}
	}
}