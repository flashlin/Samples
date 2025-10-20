using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class B2B2CRebateReportEntityConfiguration : IEntityTypeConfiguration<B2B2CRebateReportEntity>
    {
        public void Configure(EntityTypeBuilder<B2B2CRebateReportEntity> builder)
        {
            builder.ToTable("B2B2CRebateReport");

            builder.HasKey(x => x.Id);

            builder.Property(x => x.Id)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.Batch)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.Username)
                .HasColumnType("nvarchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.AgentId)
                .HasColumnType("nvarchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.MaId)
                .HasColumnType("nvarchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.SmaId)
                .HasColumnType("nvarchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.CreditDate)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.Amount)
                .HasColumnType("decimal(19,6)")
                .IsRequired()
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.AgentPt)
                .HasColumnType("decimal(3,2)")
                .IsRequired()
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.MaPt)
                .HasColumnType("decimal(3,2)")
                .IsRequired()
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.SmaPt)
                .HasColumnType("decimal(3,2)")
                .IsRequired()
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.AgentAmount)
                .HasColumnType("decimal(19,6)")
                .IsRequired()
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.MaAmount)
                .HasColumnType("decimal(19,6)")
                .IsRequired()
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.SmaAmount)
                .HasColumnType("decimal(19,6)")
                .IsRequired()
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.Currency)
                .HasColumnType("varchar(3)")
                .IsRequired()
                .HasMaxLength(3)
            ;

            builder.Property(x => x.Status)
                .HasColumnType("varchar(15)")
                .IsRequired()
                .HasMaxLength(15)
            ;

            builder.Property(x => x.CreatedOn)
                .HasColumnType("datetime")
                .IsRequired()
            ;

            builder.Property(x => x.CreatedBy)
                .HasColumnType("varchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.ModifiedOn)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.ModifiedBy)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.Remark)
                .HasColumnType("varchar(255)")
                .HasMaxLength(255)
            ;

            builder.Property(x => x.GUID)
                .HasColumnType("uniqueidentifier")
                .IsRequired()
            ;

            builder.Property(x => x.WinloseDate)
                .HasColumnType("datetime")
            ;

        }
    }
}
