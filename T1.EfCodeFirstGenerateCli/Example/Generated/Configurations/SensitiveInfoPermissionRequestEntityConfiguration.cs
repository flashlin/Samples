using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class SensitiveInfoPermissionRequestEntityConfiguration : IEntityTypeConfiguration<SensitiveInfoPermissionRequestEntity>
    {
        public void Configure(EntityTypeBuilder<SensitiveInfoPermissionRequestEntity> builder)
        {
            builder.ToTable("SensitiveInfoPermissionRequest");

            builder.HasKey(x => x.Id);

            builder.Property(x => x.Id)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.ApproverCustomerId)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.RequesterCustomerId)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.RequesterAccountId)
                .HasColumnType("nvarchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.RequestedCustomerId)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.RequestedAccountId)
                .HasColumnType("nvarchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.PageKey)
                .HasColumnType("varchar(200)")
                .IsRequired()
                .HasMaxLength(200)
            ;

            builder.Property(x => x.Reason)
                .HasColumnType("nvarchar(256)")
                .IsRequired()
                .HasMaxLength(256)
            ;

            builder.Property(x => x.RequestedOn)
                .HasColumnType("datetime")
                .IsRequired()
            ;

            builder.Property(x => x.ApprovedOn)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.UsedOn)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.Department)
                .HasColumnType("varchar(20)")
                .IsRequired()
                .HasMaxLength(20)
            ;

        }
    }
}
