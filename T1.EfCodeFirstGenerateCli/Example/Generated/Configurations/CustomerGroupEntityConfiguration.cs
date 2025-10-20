using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class CustomerGroupEntityConfiguration : IEntityTypeConfiguration<CustomerGroupEntity>
    {
        public void Configure(EntityTypeBuilder<CustomerGroupEntity> builder)
        {
            builder.ToTable("CustomerGroup");

            builder.HasKey(x => x.GroupID);

            builder.Property(x => x.GroupID)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.PGroupID)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.GroupName)
                .HasColumnType("varchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.GroupLevel)
                .HasColumnType("tinyint(3,0)")
                .IsRequired()
            ;

            builder.Property(x => x.Status)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.RiskLimit)
                .HasColumnType("decimal(19,6)")
                .IsRequired()
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

            builder.Property(x => x.Auditor)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.LastAuditOn)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.LastAuditBy)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.ExtraInfoID)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.Remark)
                .HasColumnType("varchar(255)")
                .HasMaxLength(255)
            ;

            builder.Property(x => x.SuspendLimit)
                .HasColumnType("decimal(19,6)")
            ;

            builder.Property(x => x.GroupSuspendStatus)
                .HasColumnType("bit")
            ;

            builder.Property(x => x.GroupSuspendIsEnabled)
                .HasColumnType("bit")
            ;

            builder.Property(x => x.KYCExpiryDate)
                .HasColumnType("datetime")
            ;

        }
    }
}
