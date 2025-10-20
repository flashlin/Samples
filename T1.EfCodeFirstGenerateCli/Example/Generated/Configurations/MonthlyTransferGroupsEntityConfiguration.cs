using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class MonthlyTransferGroupsEntityConfiguration : IEntityTypeConfiguration<MonthlyTransferGroupsEntity>
    {
        public void Configure(EntityTypeBuilder<MonthlyTransferGroupsEntity> builder)
        {
            builder.ToTable("MonthlyTransferGroups");

            builder.HasKey(x => x.Id);

            builder.Property(x => x.Id)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.GroupId)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.GroupName)
                .HasColumnType("nvarchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.CreatedOn)
                .HasColumnType("datetime2")
                .IsRequired()
            ;

            builder.Property(x => x.CreatedBy)
                .HasColumnType("nvarchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.ModifiedOn)
                .HasColumnType("datetime2")
            ;

            builder.Property(x => x.ModifiedBy)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.IsDeleted)
                .HasColumnType("bit")
                .IsRequired()
                .HasDefaultValue(false)
            ;

        }
    }
}
