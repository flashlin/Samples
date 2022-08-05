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
			_bindingSource.Add(new CodeSnippet { Id = 3, Content = "public class bold substring in C#" });
			//_bindingSource.DataSource = _table;

			_dataGridView.Columns.Add(new DataGridViewTextBoxColumn()
			{
				Name = "index",
				DataPropertyName = "Id",
				ReadOnly = true,
			});
			_dataGridView.Columns.Add(new DataGridViewTextBoxColumn()
			{
				Name = "Code Content",
				DataPropertyName = "content",
				ReadOnly = true,
			});

			_dataGridView.AutoGenerateColumns = false;
			_dataGridView.DataSource = _bindingSource;
			this.Controls.Add(_dataGridView);
		}
	}
}